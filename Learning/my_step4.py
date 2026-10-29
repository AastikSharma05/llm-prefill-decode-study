import csv
import statistics

prompt_lengths = [1000, 2000, 4000, 8000]
network_speeds = [10, 25, 100, 400]

groups = {}
with open('my_exp_a.csv') as f:
    for row in csv.DictReader(f):
        tokens = int(row["Tokens"])
        ttft_ms = float(row['TTFT_ms'])
        if tokens not in groups:
            groups[tokens] = []
        groups[tokens].append(ttft_ms)

median_ttft = {}
for tokens, values in groups.items():
    median_ttft[tokens] = statistics.median(values)

mb_per_token = (2*28*4*128*2)/1000000
for prompt_len in prompt_lengths:
    print(f"prompt length: {prompt_len}, notes generated: {prompt_len * mb_per_token:.2f} MB")
    for speed in network_speeds:
        usable_gbps = speed * 0.6
        transfer_ms = (prompt_len * mb_per_token * 8) / usable_gbps
        percentage_ttft = (transfer_ms / median_ttft[prompt_len] ) *100
        print(f"  network speed: {speed} Gbit/s, notes time taken: {transfer_ms:.2f} ms, percentage of TTFT: {percentage_ttft:.2f}%")
