# Key numbers (written by check_results.py)

```
## Experiment A  (100 requests, model Qwen/Qwen2.5-7B-Instruct, chunk budget 2048)
ok: every setting's TTFT repeats agree within 10%
TTFT grows 19.3x from 1k to 16k input tokens (input grew 16x)
GPU while busy (gpu_log_a.csv): SM clock median 1350 MHz, 5th pct 1065 MHz, max temperature 72 C
?? the clock dropped by more than 15% at times. The L4 is a 72 W card and lowers its clock when it hits its power limit, so some drop is normal; compare the two Exp A runs before trusting a trend.

Numbers for Figure 1 (prefill vs decode):
  median TTFT / decode time (ms):
    input              32 out              64 out             128 out             512 out
       1k       82 /     387       81 /     785       80 /    1582       81 /    6379
       2k      159 /     387      160 /     787      158 /    1588      157 /    6401
       4k      324 /     387      324 /     790      324 /    1596      325 /    6435
       8k      694 /     392      691 /     800      692 /    1618      692 /    6535
      16k     1554 /     395     1555 /     818     1546 /    1662     1558 /    6741
  - share of a 16k-input, 32-output request spent in prefill: 80% (TTFT 1554 ms, decode 395 ms)
  - prefill takes longer than decode for 32 output tokens: from 8k input tokens
  - prefill takes longer than decode for 64 output tokens: from 16k input tokens
  - prefill takes longer than decode for 128 output tokens: never, up to 16k input tokens
  - prefill takes longer than decode for 512 output tokens: never, up to 16k input tokens
  - TPOT (median over output lengths): 12.5 ms at 1k input, 13.0 ms at 16k input (+5%; each decode step reads a larger KV cache)

Rough share of the GPU's peak reached (datasheet values for A100; YOUR check: redo this arithmetic by hand):
  input   1k: prefill   54% of peak compute | decode   73% of peak memory bandwidth (floor TPOT 9.1 ms)
  input   2k: prefill   56% of peak compute | decode   73% of peak memory bandwidth (floor TPOT 9.2 ms)
  input   4k: prefill   56% of peak compute | decode   74% of peak memory bandwidth (floor TPOT 9.3 ms)
  input   8k: prefill   56% of peak compute | decode   74% of peak memory bandwidth (floor TPOT 9.4 ms)
  input  16k: prefill   55% of peak compute | decode   74% of peak memory bandwidth (floor TPOT 9.7 ms)

## Experiment A2 (estimate, model qwen2.5-7b, link efficiency 60%)
Numbers for Figure 2 (transfer vs prefill):
  - 8k input at 10 Gbit/s: adds 3.23% (ideal layer overlap, 22.37 ms) to 90.5% (no overlap, 626.3 ms) to the measured prefill of 691.8 ms
  - 8k input at 25 Gbit/s: adds 1.29% (ideal layer overlap, 8.95 ms) to 36.2% (no overlap, 250.5 ms) to the measured prefill of 691.8 ms
  - 8k input at 100 Gbit/s: adds 0.32% (ideal layer overlap, 2.24 ms) to 9.1% (no overlap, 62.6 ms) to the measured prefill of 691.8 ms
  - 8k input at 400 Gbit/s: adds 0.08% (ideal layer overlap, 0.56 ms) to 2.3% (no overlap, 15.7 ms) to the measured prefill of 691.8 ms
  - KV cache for 8k input: 469.8 MB; that is 7.34 MB shipped per output token for 64 output tokens
  - with no overlap, transfer takes as long as the measured prefill on links slower than about 9 Gbit/s (nominal); this depends on this GPU's prefill speed
  Reminder: these are estimates from KV size and link speed, not measurements. Do not call either one a bound.

## Experiment B (27 runs, model Qwen/Qwen2.5-7B-Instruct)
ok: chunk512: summarization TTFT median 864 ms colocated vs 51 ms kvremote (ratio 0.06)
ok: chunk2048: summarization TTFT median 743 ms colocated vs 52 ms kvremote (ratio 0.07)
ok: chunk8192: summarization TTFT median 704 ms colocated vs 51 ms kvremote (ratio 0.07)

Numbers for Figure 3 (interference). mean [min-max] over repeats:
  chunk512  (chat-only median gap 12.4 [12.4-12.4] ms)
    p99 gap ms       off: 28 [28-29]            colocated: 56 [56-56]            kvremote: 32 [31-32]            
    p99.9 gap ms     off: 31 [31-31]            colocated: 57 [57-57]            kvremote: 37 [37-38]            
    stall s/user-min off: 2.54 [2.44-2.61]      colocated: 22.16 [21.80-22.48]   kvremote: 3.21 [3.10-3.29]      
    -> p99 gap ms: colocated minus kvremote = 24; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> p99.9 gap ms: colocated minus kvremote = 20; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> stall s/user-min: colocated minus kvremote = 18.96; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    p99.9 of about 43546 chat gaps rests on its top ~44 gaps per run
  chunk2048  (chat-only median gap 12.4 [12.4-12.4] ms)
    p99 gap ms       off: 29 [28-29]            colocated: 177 [177-177]         kvremote: 32 [30-35]            
    p99.9 gap ms     off: 31 [31-32]            colocated: 190 [190-190]         kvremote: 38 [38-38]            
    stall s/user-min off: 2.57 [2.45-2.64]      colocated: 19.77 [19.49-20.08]   kvremote: 3.23 [3.06-3.33]      
    -> p99 gap ms: colocated minus kvremote = 145; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> p99.9 gap ms: colocated minus kvremote = 152; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> stall s/user-min: colocated minus kvremote = 16.54; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    p99.9 of about 42444 chat gaps rests on its top ~42 gaps per run
  chunk8192  (chat-only median gap 12.4 [12.4-12.4] ms)
    p99 gap ms       off: 28 [28-29]            colocated: 55 [50-66]            kvremote: 32 [30-35]            
    p99.9 gap ms     off: 31 [31-31]            colocated: 652 [651-652]         kvremote: 37 [37-38]            
    stall s/user-min off: 2.57 [2.50-2.61]      colocated: 19.34 [19.30-19.41]   kvremote: 3.22 [3.10-3.29]      
    -> p99 gap ms: colocated minus kvremote = 23; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> p99.9 gap ms: colocated minus kvremote = 614; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    -> stall s/user-min: colocated minus kvremote = 16.11; ranges do not overlap (with 3 repeats, say 'in all 3 repeats', not 'significant')
    p99.9 of about 42442 chat gaps rests on its top ~42 gaps per run

```
