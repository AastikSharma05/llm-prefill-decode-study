import json, random, statistics, threading, time, requests

name = requests.get('http://localhost:8000/v1/models')
model = name.json()["data"][0]["id"]

WITH_SUMMARY = True      # set False for run A (chat alone)

def token_gaps(prompt, max_tokens):
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
    gaps = []
    for k in range(1, len(token_times)):
        gap = token_times[k] - token_times[k-1] 
        gaps.append(gap)
    return gaps

chat_gaps = []

def chat_user():
    prompt = [random.randint(1000,20000) for _ in range(256)]
    result= token_gaps(prompt, 300)
    chat_gaps.extend(result)

def summary_customer():
    time.sleep(1)
    token_gaps([random.randint(1000,20000) for _ in range(8000)], 32)

# create a thread for chat_user
t1 = threading.Thread(target=chat_user)
if WITH_SUMMARY:
    t2 = threading.Thread(target = summary_customer)
# if WITH_SUMMARY: also create a thread for summary_customer
# start the thread(s), then join the thread(s)
t1.start();
t2.start() if WITH_SUMMARY else None
t1.join();
t2.join() if WITH_SUMMARY else None

median_gap = statistics.median(chat_gaps)
print("number of gaps:", len(chat_gaps))
print("median gap (ms):", median_gap)
print("largest gap (ms):", max(chat_gaps))
print("gaps > 3x median:", sum(1 for g in chat_gaps if g > 3 * median_gap))
print("p99 gap (ms):", statistics.quantiles(chat_gaps, n=100)[98])