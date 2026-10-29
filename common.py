"""Shared helpers: send one streaming request to a vLLM (OpenAI-compatible) server and time it.

Timing terms used everywhere in this project:
  TTFT  time to first token  = time from sending the request until the first output token arrives.
                               For a request running alone, this is almost all prefill time.
  TPOT  time per output token = (time of last token - time of first token) / (output tokens - 1).
                               This is the average decode step time.
  ITL   inter-token latency  = gap between two consecutive output tokens. Its 99th and 99.9th
                               percentiles show the "stutter" users feel when other work
                               interrupts decoding.
"""
import json
import random
import time

import aiohttp


def random_prompt(n_tokens, seed, low=1000, high=20000):
    """A prompt made of n_tokens random token IDs.

    Random IDs give an exact input length without needing the tokenizer, and every
    seed gives a different prompt, so the server's prefix cache can never reuse work
    from an earlier request (which would make prefill look faster than it is).
    IDs 1000-20000 are ordinary tokens for Qwen2.5 and Llama 3 (special tokens sit far higher).
    """
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n_tokens)]


async def get_model_name(session, base_url):
    async with session.get(f"{base_url}/v1/models") as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["data"][0]["id"]


async def stream_request(session, base_url, model, prompt_ids, max_tokens):
    """Send one request with streaming on and return its timings.

    ignore_eos=True forces exactly max_tokens output tokens, so output length is set by
    the experiment, not by the model. repetition_penalty=1.0 stops vLLM from silently
    applying the model's own default (Qwen ships 1.05), which adds work on every step.
    return_token_ids=True lets us count tokens per streamed chunk: under load the server
    can pack several tokens into one chunk, and each must be counted.
    """
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    t_send = time.perf_counter()
    chunks = []  # (arrival time, tokens in this chunk)
    usage = None
    async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"server returned {resp.status}: {body[:300]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "error" in obj:
                raise RuntimeError(f"server error mid-stream: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices"):
                ids = obj["choices"][0].get("token_ids")
                chunks.append((time.perf_counter(), len(ids) if ids else 1))
    t_done = time.perf_counter()

    if not chunks:
        raise RuntimeError("no tokens received")
    out_tokens = usage["completion_tokens"] if usage else sum(n for _, n in chunks)
    in_tokens = usage["prompt_tokens"] if usage else len(prompt_ids)

    # Per-token gaps. A chunk carrying n tokens after a gap g is what the user sees as one
    # wait of g followed by n-1 tokens arriving at once, so it counts as g, then n-1 zeros.
    token_gaps = []  # (end time of the gap, gap length in seconds)
    for (t0, _), (t1, n) in zip(chunks, chunks[1:]):
        token_gaps.append((t1, t1 - t0))
        token_gaps.extend([(t1, 0.0)] * (n - 1))

    first, last = chunks[0][0], chunks[-1][0]
    return {
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "length_ok": out_tokens == max_tokens,
        "multi_token_chunks": sum(1 for _, n in chunks if n > 1),
        "ttft_s": first - t_send,
        "decode_s": last - first,
        "tpot_s": (last - first) / (out_tokens - 1) if out_tokens > 1 else 0.0,
        "e2e_s": t_done - t_send,
        "token_gaps": token_gaps,
        "t_send": t_send,
    }


def percentile(values, p):
    """Linear-interpolation percentile (same as numpy's default). None if empty."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
