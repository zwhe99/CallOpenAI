import os
import json
import asyncio
import argparse
from transformers.utils import logging
from call_openai import CallOpenAI
from logging_utils import setup_colored_logging

setup_colored_logging()
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--request_urls", nargs="+", required=True)
    args = parser.parse_args()

    def input_to_requests_func(input_file_path: str, output_file_path: str) -> list:
        """
        Convert input file to a list of requests for OpenAI API.

        Args:
            input_file_path (str): The path to the input file.
            output_file_path (str): The path to the output file.

        Returns:
            list: A list of requests for OpenAI API.

        Note: Exclude the requests that have been done.
        """

        rqs = [] # list of requests
        done_ids = [] # list of ids that have been done

        # read the output file to get the ids that have been done
        if os.path.isfile(output_file_path):
            with open(output_file_path, "r") as f:
                for line in f:
                    done_ids.append(json.loads(line.strip())["id"])

        # read the input file to form the requests
        with open(input_file_path, "r") as f:
            for i, line in enumerate(f):
                if i in done_ids:
                    continue
                rq = {
                    "model": args.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Translate the following English text to Chinese:\n\n{line}"
                        }
                    ],
                    "metadata": {"row_id": i} # store the id of the request as metadata which will be returned by the API
                }
                rqs.append(rq)
        return rqs

    def response_to_output_func(response: dict, input_file_path: str, output_file_path: str):
        """
        Invoked after each succesful API call.
        Convert OpenAI response to output and write it to output file.

        Args:
            response (dict): Dict of OpenAI response in the format of {"response": ..., "metadata": ...}.
            input_file_path (str): The path to the input file.
            output_file_path (str): The path to the output file.

        Returns:
            None
        """
        try:
            translation = response["response"]["choices"][0]["message"]["content"] # Extract the translation from the response
            id = response["metadata"]["row_id"] # Extract the row ID from the metadata
        except Exception as e:
            logger.error(f"Convert response to output failed with error: {e}\nResponse: {json.dumps(response, indent=4, ensure_ascii=False)}")
            return

        # Write the translation to the output file as a json string. Temporarily, we use the output file as a jsonl file.
        json_string = json.dumps(
            {
                "id": id,
                "translation": translation
            },
            ensure_ascii=False
        )
        with open(output_file_path, "a") as f:
            f.write(json_string + "\n")

    def post_run_func(input_file_path: str, output_file_path: str):
        """
        Invoked after all API calls are done.
        Organize the output file into the desired format

        Args:
            output_file_path (str): The path to the output file.

        Returns:
            None
        """

        # Read the output file and sort the translations by id
        results = []
        with open(output_file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        results = sorted(results, key=lambda x: x['id'])

        # Write the translations to the output file
        translations = [r["translation"].replace("\n", " ") for r in results]
        with open(output_file_path, "w") as f:
            for t in translations:
                f.write(f"{t}\n")

    def is_all_done(input_file_path: str, output_file_path: str) -> bool:
        """
        Check if all the requests in the input file have been done.

        Args:
            input_file_path (str): The path to the input file.
            output_file_path (str): The path to the output file.

        Returns:
            bool: True if all the requests have been done, False otherwise.
        """

        if not os.path.isfile(output_file_path):
            return False

        with open(input_file_path, "r") as f:
            num_requests = len(f.readlines())

        with open(output_file_path, "r") as f:
            num_done = len(f.readlines())

        return num_requests == num_done

    openai_caller = CallOpenAI(
        request_urls=args.request_urls,
        api_key="dummy",
        input_file_path="wmt22.en-zh.en",
        output_file_path="wmt22.en-zh.zh",
        max_attempts=5,
        max_connections=1000,
        seconds_to_sleep_each_loop=1e-5,

        # Set the functions for converting input to requests, converting response to output, running after all API calls are done, and checking if all requests have been done
        input_to_requests_func=input_to_requests_func,
        response_to_output_func=response_to_output_func,
        post_run_func=post_run_func,
        is_all_done_func=is_all_done,

        logging_level=logging.INFO
    )

    asyncio.run(
        openai_caller.run()
    )
