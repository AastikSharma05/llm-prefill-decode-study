"""Experiment A2: what would it cost to ship a request's KV cache to a separate decode GPU?

No GPU needed. Uses the model's shape to compute the KV cache size, then estimates the
transfer time over a network link. With disaggregation, decode still has to run on the
decode GPU afterwards, so the transfer does not replace any work: it is EXTRA latency
before the first token (added TTFT), plus network traffic and memory on the decode GPU.

    KV bytes per token = 2 (K and V) x layers x KV heads x head dim x bytes per value

Two estimates bracket the added TTFT:
  no overlap      the whole cache is sent after prefill finishes:  t = bytes / (link x efficiency)
  layer overlap   each layer is sent while the next layer computes (as in DistServe and
                  DejaVu): exposed time = max(t / layers, t - prefill x (layers - 1) / layers)
Real systems land between the two, and per-message overheads can push them above "no overlap".
"efficiency" (default 0.6) accounts for links rarely reaching their nominal line rate.

Examples:
  python exp_a2_kv_model.py                                   # KV sizes only
  python exp_a2_kv_model.py --exp-a results/exp_a.csv --model-key qwen2.5-7b
  python exp_a2_kv_model.py --config-json path/to/config.json # any HF model
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

# Shapes taken from each model's config.json on Hugging Face.
MODELS = {
    "qwen2.5-1.5b": {"layers": 28, "kv_heads": 2, "head_dim": 128},
    "qwen2.5-7b": {"layers": 28, "kv_heads": 4, "head_dim": 128},
    "llama-3.1-8b": {"layers": 32, "kv_heads": 8, "head_dim": 128},
    "llama-3.1-70b": {"layers": 80, "kv_heads": 8, "head_dim": 128},
}
BANDWIDTHS_GBPS = [10, 25, 100, 400]  # nominal gigabits per second


def kv_bytes_per_token(shape, bytes_per_value=2):
    return 2 * shape["layers"] * shape["kv_heads"] * shape["head_dim"] * bytes_per_value


def shape_from_config(path):
    with open(path) as f:
        c = json.load(f)
    heads = c["num_attention_heads"]
    return {
        "layers": c["num_hidden_layers"],
        "kv_heads": c.get("num_key_value_heads", heads),
        "head_dim": c.get("head_dim") or c["hidden_size"] // heads,
    }


def transfer_seconds(n_bytes, gbps, efficiency):
    return n_bytes / (gbps * 1e9 / 8 * efficiency)


def guess_key(model_name):
    name = model_name.lower()  # e.g. "qwen/qwen2.5-7b-instruct"
    for key in MODELS:
        if key in name:
            return key
    return None


def main(args):
    models = dict(MODELS)
    if args.config_json:
        models["custom"] = shape_from_config(args.config_json)
    eff = args.efficiency

    print(f"KV cache size and full transfer time for one 16,384-token input "
          f"(link efficiency {eff:.0%}, no overlap)\n")
    print(f"{'model':<15}{'kB/token':>10}{'MB @16k':>10}" + "".join(f"{str(g) + ' Gb/s':>12}" for g in BANDWIDTHS_GBPS))
    for key, shape in models.items():
        per_tok = kv_bytes_per_token(shape, args.bytes_per_value)
        total = per_tok * 16384
        cells = "".join(f"{transfer_seconds(total, g, eff) * 1000:>9.0f} ms" for g in BANDWIDTHS_GBPS)
        print(f"{key:<15}{per_tok / 1e3:>10.1f}{total / 1e6:>10.0f}{cells}")

    if not args.exp_a:
        return

    with open(args.exp_a) as f:
        rows = list(csv.DictReader(f))
    key = args.model_key or guess_key(rows[0]["model"])
    if key not in models:
        raise SystemExit(f"cannot match model '{rows[0]['model']}' to a shape; pass --model-key or --config-json")
    shape = models[key]
    per_tok = kv_bytes_per_token(shape, args.bytes_per_value)

    # Prefill time does not depend on output length, so pool all outputs per input length.
    ttft_by_input = defaultdict(list)
    out_lens = sorted({int(r["output_len"]) for r in rows})
    for r in rows:
        ttft_by_input[int(r["input_len"])].append(float(r["ttft_s"]))

    out_rows = []
    for in_len in sorted(ttft_by_input):
        prefill = statistics.median(ttft_by_input[in_len])
        kv = per_tok * in_len
        for gbps in BANDWIDTHS_GBPS:
            t_full = transfer_seconds(kv, gbps, eff)
            # With layer overlap, layer i is sent while layer i+1 computes. Only the last
            # layer is exposed, unless the link is slower than compute and falls behind.
            L = shape["layers"]
            t_overlap = max(t_full / L, t_full - prefill * (L - 1) / L)
            out_rows.append({
                "model_key": key,
                "input_len": in_len,
                "bandwidth_gbps": gbps,
                "link_efficiency": eff,
                "kv_mb": round(kv / 1e6, 1),
                "prefill_ms": round(prefill * 1000, 1),
                "transfer_no_overlap_ms": round(t_full * 1000, 1),
                "transfer_layer_overlap_ms": round(t_overlap * 1000, 2),
                "added_ttft_pct_no_overlap": round(100 * t_full / prefill, 1),
                "added_ttft_pct_layer_overlap": round(100 * t_overlap / prefill, 2),
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nAdded TTFT from shipping the KV cache, as % of the measured prefill time ({key}).")
    print("Each cell: no overlap / ideal layer overlap\n")
    print(f"{'input':>7}{'prefill':>11}" + "".join(f"{str(g) + ' Gb/s':>18}" for g in BANDWIDTHS_GBPS))
    for in_len in sorted(ttft_by_input):
        rs = [r for r in out_rows if r["input_len"] == in_len]
        cells = "".join(f"{r['added_ttft_pct_no_overlap']:>9}% /{r['added_ttft_pct_layer_overlap']:>6}%" for r in rs)
        print(f"{in_len:>7}{rs[0]['prefill_ms']:>8.0f} ms{cells}")

    print("\nKV megabytes shipped per generated output token (the summarization asymmetry)\n")
    print(f"{'input':>7}" + "".join(f"{str(o) + ' out':>11}" for o in out_lens))
    for in_len in sorted(ttft_by_input):
        print(f"{in_len:>7}" + "".join(f"{per_tok * in_len / o / 1e6:>11.2f}" for o in out_lens))
    print(f"\nsaved {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-a", default=None, help="CSV written by exp_a_sweep.py")
    p.add_argument("--model-key", default=None, choices=list(MODELS) + ["custom"])
    p.add_argument("--config-json", default=None, help="a Hugging Face config.json to add as 'custom'")
    p.add_argument("--bytes-per-value", type=int, default=2, help="2 for fp16/bf16, 1 for fp8")
    p.add_argument("--efficiency", type=float, default=0.6, help="share of nominal link speed achieved")
    p.add_argument("--out", default="results/exp_a2_transfer.csv")
    main(p.parse_args())
