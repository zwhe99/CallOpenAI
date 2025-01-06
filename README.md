##### Start the Demo

```shell
python3 run.py --model_name xxx --request_urls http://xxx.xxx.xxx.xxx:8000/v1/chat/completions
```

* See `run.py` for usage



##### Default Rate Limit (per minute per URL)

```python
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
```



**Maximize the Throughput**

1. Set debug mode: `logging_level=logging.DEBUG`

2. Tune `max_connections` and `seconds_to_sleep_each_loop` so that the printed speed (requests/min and tokens/min) reach set rate limit.

   *Note: The debug log prints overall speed. `MODEL2RPM` and `MODEL2TPM` are speed limit per URL.*

3. Set `MODEL2RPM` and `MODEL2TPM` to appropriate numbers so that the server has a certain amount of pending requests. (or just set them according to user tier)
4. Repeat 2 and 3.

 