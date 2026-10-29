import json
import requests
import time
import random
import statistics
import csv

name = requests.get('http://localhost:8000/v1/models')
model = name.json()["data"][0]["id"]
lengths = [1000, 2000, 4000, 8000]
output_lengths = [32, 128]

def measure(prompt, max_tokens):
    start = time.perf_counter()*1000
    response = requests.post('http://localhost:8000/v1/completions', json= {"prompt":prompt, "max_tokens": max_tokens, "model":model}, stream=True)
    token_times= []
    for line in response.iter_lines():
        now = time.perf_counter()*1000
        if not line:
            continue
        text = line.decode()
        if text == 'data: [DONE]':
            break
        obj = json.loads(text[6:])
        if obj.get('choices'):
            token_times.append(now)
    ttft = token_times[0] - start
    if len(token_times) <= 1:
        tpot = None
    else: 
        tpot = (token_times[-1] - token_times[0])/ (len(token_times)-1)
    return ttft, tpot

# Warm up
n = 500
measure([random.randint(1000,20000) for _ in range(n)],32)

with open("my_exp_a.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration","Tokens", "OutputLength", "TTFT_ms", "TPOT_ms"])
    for n in lengths:
        for j in output_lengths:
            median_ttft = []
            median_tpot = []
            for i in range(3):
                ttft, tpot = measure([random.randint(1000,20000) for _ in range(n)], j)
                writer.writerow([i+1, n, j, round(ttft,2), round(tpot,2)])
                median_ttft.append(ttft)
                median_tpot.append(tpot)
            print(f"Tokens: {n}, OutputLength: {j}")
            print(f"Median TTFT: {statistics.median(median_ttft)} ms")
            print(f"Median TPOT: {statistics.median(median_tpot)} ms")