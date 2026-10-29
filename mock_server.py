"""A FAKE LLM server for practising on a laptop with no GPU.

It speaks the same streaming API as vLLM and roughly imitates how vLLM's V1 scheduler
fills each step's token budget (--max-num-batched-tokens), including prefix caching and
--long-prefill-token-threshold. That is enough for every script in this project to run
end to end, so you can debug before you pay for a GPU. Its timing model is a toy: how
real vLLM behaves is exactly what the experiments are for.

THE NUMBERS IT PRODUCES ARE INVENTED. Its model name is "MOCK-not-real-data" so mock
results are easy to spot. Never put them in a report or the proposal.

  python mock_server.py --port 8000 --max-num-batched-tokens 2048
"""
import argparse
import asyncio
import json
import time

from aiohttp import web

MODEL_NAME = "MOCK-not-real-data"


class Req:
    def __init__(self, n_prompt, n_out, cached):
        self.n_prompt = n_prompt
        # A prefix-cache hit leaves only the last partial block (here 16 tokens) to compute.
        self.prefill_left = 16 if cached else n_prompt
        self.out_left = n_out
        self.queue = asyncio.Queue()


class Engine:
    """Imitates vLLM V1 scheduling: requests are served in admission order; each takes
    min(what it still needs, what is left of the step's token budget); once the budget is
    used up, later requests wait for the next step, even if they only need one decode token.
    """

    def __init__(self, budget, long_prefill_threshold, base_ms, prefill_us_per_tok,
                 decode_us_per_seq, decode_us_per_1k_ctx, prefix_caching):
        self.budget = budget
        self.threshold = long_prefill_threshold
        self.base = base_ms / 1000
        self.p_cost = prefill_us_per_tok / 1e6
        self.d_cost = decode_us_per_seq / 1e6
        self.ctx_cost = decode_us_per_1k_ctx / 1e6
        self.prefix_caching = prefix_caching
        self.seen = set()
        self.reqs = []  # admission order
        self.wake = asyncio.Event()

    def add(self, prompt, n_out):
        key = hash(tuple(prompt)) if isinstance(prompt, list) else hash(prompt)
        cached = self.prefix_caching and key in self.seen
        self.seen.add(key)
        n_prompt = len(prompt) if isinstance(prompt, list) else max(1, len(prompt) // 4)
        req = Req(n_prompt, n_out, cached)
        self.reqs.append(req)
        self.wake.set()
        return req

    async def loop(self):
        while True:
            if not self.reqs:
                self.wake.clear()
                await self.wake.wait()
            budget = self.budget
            prefill_tokens, decode_seqs, decode_ctx = 0, [], 0
            progress = []
            for r in self.reqs:
                if budget <= 0:
                    break
                if r.prefill_left > 0:
                    need = r.prefill_left if self.threshold <= 0 else min(r.prefill_left, self.threshold)
                    take = min(need, budget)
                    prefill_tokens += take
                    budget -= take
                    progress.append((r, take))
                else:
                    budget -= 1
                    decode_seqs.append(r)
                    decode_ctx += r.n_prompt
            step = (self.base + prefill_tokens * self.p_cost + len(decode_seqs) * self.d_cost
                    + decode_ctx / 1000 * self.ctx_cost)
            await asyncio.sleep(step)
            done = []
            for r, take in progress:
                r.prefill_left -= take
                if r.prefill_left == 0:  # the step that finishes prefill also emits the first token
                    decode_seqs.append(r)
            for r in decode_seqs:
                r.out_left -= 1
                r.queue.put_nowait("tok")
                if r.out_left == 0:
                    r.queue.put_nowait(None)
                    done.append(r)
            for r in done:
                self.reqs.remove(r)


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})


async def health(request):
    return web.Response(text="")


async def completions(request):
    body = await request.json()
    prompt = body["prompt"]
    n_out = int(body.get("max_tokens", 16))
    req = request.app["engine"].add(prompt, n_out)
    n_prompt = req.n_prompt

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    created = int(time.time())
    while True:
        item = await req.queue.get()
        if item is None:
            break
        chunk = {"id": "mock", "object": "text_completion", "created": created, "model": MODEL_NAME,
                 "choices": [{"index": 0, "text": " x", "token_ids": [1], "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}
    final = {"id": "mock", "object": "text_completion", "created": created, "model": MODEL_NAME,
             "choices": [], "usage": usage}
    await resp.write(f"data: {json.dumps(final)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def start_engine(app):
    app["engine_task"] = asyncio.create_task(app["engine"].loop())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--long-prefill-token-threshold", type=int, default=0, help="0 = off")
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--base-ms", type=float, default=8.0, help="fixed cost of one step")
    p.add_argument("--prefill-us-per-token", type=float, default=20.0)
    p.add_argument("--decode-us-per-seq", type=float, default=150.0)
    p.add_argument("--decode-us-per-1k-ctx", type=float, default=40.0)
    a = p.parse_args()
    app = web.Application()
    app["engine"] = Engine(a.max_num_batched_tokens, a.long_prefill_token_threshold, a.base_ms,
                           a.prefill_us_per_token, a.decode_us_per_seq, a.decode_us_per_1k_ctx,
                           a.enable_prefix_caching)
    app.on_startup.append(start_engine)
    app.add_routes([web.get("/v1/models", models), web.get("/health", health),
                    web.post("/v1/completions", completions)])
    print(f"MOCK server (fake numbers) on port {a.port}, step budget {a.max_num_batched_tokens} tokens")
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
