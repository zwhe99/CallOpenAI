import os        # for reading API key
import re        # for matching endpoint from request URL
import sys       # for reconfiguring stdout and stderr
import time      # for sleeping after rate limit is hit
import asyncio   # for running API calls concurrently
import itertools # for cycling through request URLs

from typing import Callable
from functools import lru_cache
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from transformers.utils import logging

import aiohttp   # for making API calls concurrently
import tiktoken  # for counting tokens
from tqdm import tqdm

logger = logging.get_logger(__name__)

# Constants: max requests and tokens per minute per URL
# Should be set according to usage tier
MODEL2RPM = {
    "gpt-4o":                            295680,
    "gpt-4o-mini":                       295680,
    "meta-llama/Llama-3.3-70B-Instruct": 20000,
}

MODEL2TPM = {
    "gpt-4o":                             29568000,
    "gpt-4o-mini":                        29568000,
    "meta-llama/Llama-3.3-70B-Instruct":  2000000,
}

# price per token in USD
PRICE_PER_INPUT_TOKEN = {
    "gpt-4o":        2.5 / 1000000,
    "gpt-4o-mini": 0.150 / 1000000,
    "meta-llama/Llama-3.3-70B-Instruct":  0,
}

PRICE_PER_OUTPUT_TOKEN = {
    "gpt-4o":       10.0 / 1000000,
    "gpt-4o-mini": 0.600 / 1000000,
    "meta-llama/Llama-3.3-70B-Instruct":  0,
}

# reconfigure stdout and stderr to be line-buffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    try:
        match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
        if match is None:
            # for Azure OpenAI deployment urls
            match = re.search(
                r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
            )
        return match[1]
    except Exception as e:
        logger.warning_once(f"Failed to extract API endpoint from {request_url}. Using default endpoint: chat/completions")
        return "chat/completions"

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

@lru_cache(maxsize=None)
def get_encoding(model: str):
    """Get the encoding for a model. If the model is not found in tiktoken, use gpt-4o as fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning_once(f"Model {model} not found in tiktoken. Using gpt-4o as fallback.") #! Do NOT use transformers.AutoTokenizer which is very slow
        return tiktoken.encoding_for_model("gpt-4o")

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    model: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = get_encoding(model)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_total: int = 0
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    num_requests_sent: int = 0
    num_tokens_sent: int = 0

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    response_to_output_func: Callable[[dict, str, str], None]
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        input_filepath: str,
        save_filepath: str,
        status_tracker: StatusTracker,
        progress_bar: tqdm
    ):
        """Calls the OpenAI API and saves results."""
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                if response.content_type == "text/plain":
                    # Should be json, but sometimes it's text which means error
                    response = await response.text()
                    response = {"error": {"message": response}}
                else:
                    response = await response.json()

            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

            if  "detail" in response and "not found" in response["detail"].lower():
                status_tracker.num_other_errors += 1
                error = response

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            if isinstance(error, TimeoutError):
                # TODO: test the effect of this
                # Vllm server would not return a RateLimitError, so we treat TimeoutError as RateLimitError
                # status_tracker.time_of_last_rate_limit_error = time.time()
                # status_tracker.num_rate_limit_errors += 1
                # status_tracker.num_api_errors -= 1  # rate limit errors are counted separately
                pass
            else:
                status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left > 0:
                if not isinstance(self.result[-1], TimeoutError):
                    logger.warning(f"""=====
Request {self.task_id} failed. Retry attempt {self.attempts_left} left.
    url: {request_url}
    error: {repr(self.result[-1])}
""")
                retry_queue.put_nowait(self) # Put the request back into the queue if attempts_left > 0
            else:
                logger.error(f"""=====
Request {self.task_id} failed. Retry attempt {self.attempts_left} left.
    url: {request_url}
    error: {repr(self.result[-1])}
""")
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = {"response": response, "metadata": self.metadata if self.metadata else None}
            self.response_to_output_func(data, input_filepath, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            progress_bar.update(n=1)

class CallOpenAI:
    def __init__(
        self,
        request_urls,                                 # list of request URLs
        api_key,                                      # API key
        input_file_path,                              # input file path
        output_file_path,                             # output file path
        input_to_requests_func,                       # function to convert input file to requests
        response_to_output_func,                      # function to convert response to output
        is_all_done_func=None,                        # function to check if all tasks are done
        post_run_func=None,                           # function to run after all tasks are done
        max_attempts=5,                               # maximum number of attempts to retry a request
        max_connections=100,                          # maximum number of connections 
        seconds_to_pause_after_rate_limit_error=15,   # seconds to pause after rate limit error
        seconds_to_sleep_each_loop=1e-3,              # seconds to sleep each loop
        progress_bar_desc=None,                       # description of the progress bar
        logging_level=logging.INFO                    # logging level
    ):
        self.num_request_urls = len(request_urls)
        self.request_urls = itertools.cycle(request_urls)
        self.api_key = api_key
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_to_requests_func = input_to_requests_func
        self.response_to_output_func = response_to_output_func
        self.is_all_done_func = is_all_done_func
        self.post_run_func = post_run_func
        self.api_endpoint = api_endpoint_from_url(next(self.request_urls))
        self.max_attempts = max_attempts
        self.max_connections = max_connections
        self.seconds_to_pause_after_rate_limit_error = seconds_to_pause_after_rate_limit_error
        self.seconds_to_sleep_each_loop = seconds_to_sleep_each_loop
        self.logging_level = logging_level
        self.progress_bar = tqdm(desc=progress_bar_desc)

        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
        if "/deployments" in next(self.request_urls):
            # use api-key header for Azure deployments
            self.request_header = {"api-key": f"{self.api_key}"}

        # initialize logging
        logger.setLevel(self.logging_level)
        logging.set_verbosity(self.logging_level)

        # initialize trackers
        self.queue_of_requests_to_retry = asyncio.Queue()
        self.task_id_generator = task_id_generator_function()  # generates integer IDs of 0, 1, 2, ...
        self.status_tracker = StatusTracker()  # single instance to track a collection of variables
        self.next_request = None  # variable to hold the next request to call

        # initialize available capacity counts
        self.model = None
        self.max_requests_per_minute = None
        self.max_tokens_per_minute = None
        self.available_request_capacity = None
        self.available_token_capacity = None
        self.last_update_time = time.time()

        # initialize flags
        self.file_not_finished = True  # after file is empty, we'll skip reading it

        # Check input & output file
        assert os.path.isfile(self.input_file_path), f"Input file {self.input_file_path} does NOT exist or is a dir."

        output_directory = os.path.dirname(self.output_file_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        logger.debug(f"Initialization complete.")

    def set_input_to_requests_func(self, input_to_requests_func):
        self.input_to_requests_func = input_to_requests_func

    def set_response_to_output_func(self, response_to_output_func):
        self.response_to_output_func = response_to_output_func

    async def run(self):
        if self.is_all_done_func is not None and self.is_all_done_func(self.input_file_path, self.output_file_path):
            logger.info("All done!")
            return

        requests = self.input_to_requests_func(self.input_file_path, self.output_file_path)

        # set progress bar
        total = len(requests)
        self.progress_bar.reset(total=total)
        self.status_tracker.num_tasks_total = total

        # set iterator
        requests = iter(requests)

        # set session
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_connections)) as session:  # Initialize ClientSession here
            start_time = time.time()
            while True:
                # get next request (if one is not already waiting for capacity)
                if self.next_request is None:
                    if not self.queue_of_requests_to_retry.empty():
                        self.next_request = self.queue_of_requests_to_retry.get_nowait()
                        logger.debug(f"Retrying request {self.next_request.task_id}")
                    elif self.file_not_finished:
                        try:
                            # get new request
                            request_json = next(requests)
                            assert "model" in request_json, "`model` is required in request"
                            if self.model is None:
                                self.model = request_json["model"]
                                self.max_requests_per_minute = MODEL2RPM[self.model] * self.num_request_urls
                                self.max_tokens_per_minute = MODEL2TPM[self.model] * self.num_request_urls
                                self.available_request_capacity = self.max_requests_per_minute
                                self.available_token_capacity = self.max_tokens_per_minute

                            # get num_tokens_consumed
                            #? might be better to if we use `usage` from the response instead of computing it locally?
                            #? but Semaphore would be slow
                            num_tokens_consumed = request_json.get("num_tokens_consumed", None)
                            if num_tokens_consumed is not None:
                                request_json.pop("num_tokens_consumed")
                            else:
                                num_tokens_consumed = num_tokens_consumed_from_request(request_json, self.api_endpoint, self.model)

                            # create next request
                            self.next_request = APIRequest(
                                task_id=next(self.task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed,
                                attempts_left=self.max_attempts,
                                metadata=request_json.pop("metadata", None),
                                response_to_output_func=self.response_to_output_func
                            )

                            # update status tracker
                            self.status_tracker.num_tasks_started += 1
                            self.status_tracker.num_tasks_in_progress += 1
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logger.debug("Read file exhausted")
                            self.file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - self.last_update_time
                assert self.max_requests_per_minute is not None, "max_requests_per_minute is not set. This shoud not happen."
                assert self.max_tokens_per_minute is not None, "max_tokens_per_minute is not set. This shoud not happen."
                before_request_capacity = self.available_request_capacity
                before_token_capacity = self.available_token_capacity
                self.available_request_capacity = min(
                    self.available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute,
                )
                self.available_token_capacity = min(
                    self.available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute,
                )
                self.last_update_time = current_time
                logger.debug(f"Updated available capacity. Request capacity: {before_request_capacity} -> {self.available_request_capacity}, Token capacity: {before_token_capacity} -> {self.available_token_capacity}")

                # speed per minute
                logger.debug(f"Speed: {self.status_tracker.num_requests_sent / (time.time() - start_time) * 60} requests/min, {self.status_tracker.num_tokens_sent / (time.time() - start_time) * 60} tokens/min")

                # if enough capacity available, call API
                if self.next_request:
                    next_request_tokens = self.next_request.token_consumption
                    if self.available_request_capacity >= 1 and self.available_token_capacity >= next_request_tokens:
                        # update counters
                        self.available_request_capacity -= 1
                        self.available_token_capacity -= next_request_tokens
                        self.next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            self.next_request.call_api(
                                session=session,
                                request_url=next(self.request_urls),
                                request_header=self.request_header,
                                retry_queue=self.queue_of_requests_to_retry,
                                input_filepath=self.input_file_path,
                                save_filepath=self.output_file_path,
                                status_tracker=self.status_tracker,
                                progress_bar=self.progress_bar
                            )
                        )
                        self.status_tracker.num_requests_sent += 1
                        self.status_tracker.num_tokens_sent += next_request_tokens
                        self.next_request = None  # reset next_request to empty
                    else:
                        logger.debug(f"Not enough capacity to call API. Available request capacity: {self.available_request_capacity}, Available token capacity: {self.available_token_capacity}, Next request tokens: {next_request_tokens}")

                # if all tasks are finished, break
                if self.status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(self.seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = time.time() - self.status_tracker.time_of_last_rate_limit_error
                if seconds_since_rate_limit_error < self.seconds_to_pause_after_rate_limit_error:
                    remaining_seconds_to_pause = self.seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                    await asyncio.sleep(remaining_seconds_to_pause)
                    logger.warning(f"Pausing to cool down for {remaining_seconds_to_pause} seconds")

        if self.status_tracker.num_tasks_total == self.status_tracker.num_tasks_succeeded:
            if self.post_run_func is not None:
                self.post_run_func(self.input_file_path, self.output_file_path)
            logger.info("All done!")
        else:
            assert self.status_tracker.num_tasks_failed == (self.status_tracker.num_tasks_total - self.status_tracker.num_tasks_succeeded)
            logger.info(f"{self.status_tracker.num_tasks_failed} tasks failed.")
