#!/usr/bin/env python3
"""
LLM inference benchmark: Qwen2.5-7B on a single NVIDIA A100-SXM4-80GB via vLLM.

Measures the two phases of autoregressive inference separately and checks both
against the hardware roofline:

  prefill  - parallel processing of the prompt   (compute-bound, GEMM)
  decode   - sequential token generation         (bandwidth-bound, GEMV)

Usage:  python bench.py [--out results]

See README.md for results and interpretation.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# ---- model + hardware ------------------------------------------------------
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 8192

# Qwen2.5-7B architecture (from config.json), needed for the roofline model.
PARAMS = 7.615e9
HIDDEN = 3584
LAYERS = 28
KV_HEADS = 4
HEAD_DIM = 128

# A100-SXM4-80GB: 2039 GB/s HBM2e, 312 TFLOP/s dense bf16 (no sparsity).
PEAK_BW = 2039e9
PEAK_FLOPS = 312e12


def build_engine():
    return LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",       # sm80+ has native bf16
        max_model_len=MAX_LEN,
        gpu_memory_utilization=0.90,
        enforce_eager=True,     # no CUDA-graph replay -> latency reflects real kernels
        seed=1234,
    )


def kv_capacity(llm):
    """Total KV tokens the engine actually allocated, read from the live engine."""
    for path in ("llm_engine.cache_config", "llm_engine.model_config.cache_config"):
        obj = llm
        try:
            for key in path.split("."):
                obj = getattr(obj, key)
            blocks = getattr(obj, "num_gpu_blocks", None)
            if blocks:
                return blocks * obj.block_size
        except AttributeError:
            continue
    return None


def exact_prompt(n_tokens, seed=0):
    """Token-exact prompt. String repetition drifts from the intended length."""
    rng = np.random.default_rng(seed)
    return TokensPrompt(prompt_token_ids=rng.integers(1000, 150000, n_tokens).tolist())


def measure(llm, prompt, max_tokens, seed=0):
    """TTFT and ITL via subtraction.

    vLLM's offline LLM.generate returns one final result per prompt, so there is
    no per-token stream to time directly. Instead: run once asking for 1 token
    (total == prefill), then once asking for N tokens (total == prefill + decode).
    The difference is the decode phase. Both runs use the same prompt, so the
    prefill cost is identical.
    """
    t0 = time.perf_counter()
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True))
    ttft_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(max_tokens=max_tokens,
                                                temperature=0.0, ignore_eos=True))
    total_ms = (time.perf_counter() - t0) * 1e3

    n_out = len(out[0].outputs[0].token_ids)
    n_in = len(out[0].prompt_token_ids)
    decode_ms = total_ms - ttft_ms
    return dict(
        prompt_tokens=n_in,
        output_tokens=n_out,
        ttft_ms=ttft_ms,
        decode_ms=decode_ms,
        itl_mean_ms=decode_ms / max(n_out - 1, 1),
        decode_tok_s=(n_out - 1) / (decode_ms / 1e3),
    )


def sweep_prefill(llm):
    rows = []
    for n in [128, 256, 512, 1024, 2048, 4096, 8192]:
        r = measure(llm, exact_prompt(n), 1)
        # A 1-token run has no decode phase, so ITL columns are undefined here.
        for k in ("decode_ms", "itl_mean_ms", "decode_tok_s"):
            r[k] = np.nan
        r["experiment"] = "prefill"
        rows.append(r)
        print(f"  n={n:5d}  TTFT={r['ttft_ms']:7.1f} ms  ({r['ttft_ms']*1e6/n:6.0f} ns/token)")
    return pd.DataFrame(rows)


def sweep_decode(llm):
    rows = []
    for n in [16, 64, 128, 256, 512]:
        r = measure(llm, exact_prompt(512), n)
        r["experiment"] = "decode"
        rows.append(r)
        print(f"  out={n:4d}  ITL={r['itl_mean_ms']:6.2f} ms  {r['decode_tok_s']:6.1f} tok/s")
    return pd.DataFrame(rows)


def sweep_concurrency(llm, prompt_tokens=512, output_tokens=256):
    rows = []
    for c in [1, 2, 4, 8, 16, 32, 64]:
        # Distinct prompts per slot: identical prompts would be served from the
        # prefix cache and would not measure real scheduling work.
        prompts = [exact_prompt(prompt_tokens, seed=200 + i) for i in range(c)]
        t0 = time.perf_counter()
        out = llm.generate(prompts, SamplingParams(max_tokens=output_tokens,
                                                   temperature=0.0, ignore_eos=True),
                           use_tqdm=False)
        wall = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in out)
        rows.append(dict(concurrency=c, total_tokens=toks, wall_s=wall,
                         aggregate_tok_s=toks / wall, per_user_tok_s=toks / wall / c))
        print(f"  c={c:3d}  aggregate={toks/wall:8.1f} tok/s  per-user={toks/wall/c:6.2f} tok/s")
    return pd.DataFrame(rows)


def roofline_report(prefill, decode, throughput, weights_gib):
    ridge = PEAK_FLOPS / PEAK_BW
    print(f"\nridge point                {ridge:8.1f} FLOP/byte")
    print(f"decode intensity            {2*PARAMS/(2*PARAMS):8.2f} FLOP/byte "
          f"({100*2*PARAMS/(2*PARAMS)/ridge:.2f}% of ridge -> memory-bound)")

    print("\nprefill, with causal attention (2*n^2*d*L per layer):")
    for n in (512, 2048, 4096, 8192):
        ttft = prefill.ttft_ms[prefill.prompt_tokens == n].iloc[0]
        gemm = 2 * PARAMS * n
        attn = 2 * n * n * HIDDEN * LAYERS     # causal: half the score matrix
        flops = gemm + attn
        print(f"  n={n:5d}  intensity {flops/(2*PARAMS):8.0f} FLOP/byte  "
              f"attention {100*attn/flops:5.1f}% of FLOPs  "
              f"MFU {100*flops/(ttft/1e3)/PEAK_FLOPS:5.1f}%")

    itl_s = decode.itl_mean_ms.iloc[-1] / 1e3
    weights = weights_gib * 2**30
    roofline_itl = weights / PEAK_BW
    achieved_bw = PARAMS * 2 / itl_s          # 2 bytes per bf16 parameter
    print(f"\nweights resident            {weights_gib:8.2f} GiB")
    print(f"ITL roofline               {roofline_itl*1e3:8.2f} ms  ({1/roofline_itl:.0f} tok/s ceiling)")
    print(f"measured, batch 1          {itl_s*1e3:8.2f} ms  ({1/itl_s:.0f} tok/s)")
    print(f"achieved bandwidth         {100*achieved_bw/PEAK_BW:8.1f}% of peak")

    row = throughput[throughput.concurrency == throughput.concurrency.max()].iloc[0]
    steps_per_s = row.total_tokens / row.concurrency / row.wall_s
    kv_bytes = row.concurrency * (512 + 128) * 2 * LAYERS * KV_HEADS * HEAD_DIM * 2
    bytes_per_s = (weights + kv_bytes) * steps_per_s
    flops_per_s = 2 * PARAMS * row.concurrency * steps_per_s
    print(f"\nat c={int(row.concurrency)}: {steps_per_s:.0f} decode steps/s")
    print(f"  bandwidth {bytes_per_s/1e9:6.0f} GB/s = {100*bytes_per_s/PEAK_BW:.0f}% of peak")
    print(f"  compute   {flops_per_s/1e12:6.0f} TFLOP/s = {100*flops_per_s/PEAK_FLOPS:.0f}% of peak")
    print("  -> neither roof saturated; the limit is per-step overhead, not the GPU")


def plot_all(out, prefill, decode, throughput):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(prefill.prompt_tokens, prefill.ttft_ms, "o-")
    ax[0].axhline(30, ls=":", c="gray", lw=1, label="~30 ms fixed overhead")
    ax[0].set(xlabel="prompt tokens", ylabel="TTFT (ms)",
              title="Prefill: TTFT vs context (compute-bound)")
    ax[0].legend(fontsize=8); ax[0].set_xscale("log", base=2)

    ax[1].plot(decode.output_tokens, decode.itl_mean_ms, "s-")
    ax[1].axhline(decode.itl_mean_ms.iloc[-1], ls=":", c="gray", lw=1)
    ax[1].set(xlabel="output tokens", ylabel="ITL (ms/token)",
              title="Decode: ITL vs output length (bandwidth-bound)")
    ax[1].set_ylim(0, 16)

    ax[2].plot(throughput.concurrency, throughput.aggregate_tok_s, "o-", label="aggregate")
    ax[2].plot(throughput.concurrency, throughput.per_user_tok_s, "s--", label="per user")
    ax[2].axvline(16, ls=":", c="r", lw=1, label="saturation knee")
    ax[2].set(xlabel="concurrency", ylabel="tokens/sec", title="Throughput vs concurrency")
    ax[2].legend(fontsize=8); ax[2].set_xscale("log", base=2)

    plt.tight_layout(); plt.savefig(out / "figures.png", dpi=150)

    ridge = PEAK_FLOPS / PEAK_BW
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ai = np.logspace(0, np.log10(ridge * 2), 100)
    ax.plot(ai, PEAK_BW * ai, c="k", lw=1.6, label=f"compute roof ({PEAK_FLOPS/1e12:.0f} TFLOP/s)")
    ax.plot(ai, np.full_like(ai, PEAK_BW), c="k", ls="--", lw=1.6,
            label=f"memory roof ({PEAK_BW/1e9:.0f} GB/s)")
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.annotate(f"ridge {ridge:.0f} FLOP/byte", xy=(ridge, 4e13),
                xytext=(ridge * 1.3, 4e13), fontsize=8, color="gray")

    itl = decode.itl_mean_ms.iloc[-1] / 1e3
    ax.scatter([2 * PARAMS / (2 * PARAMS)], [PARAMS * 2 / itl], s=160, c="red", zorder=5,
               label=f"decode b=1: {itl*1e3:.2f} ms/tok\n{PARAMS*2/itl/1e9:.0f} GB/s "
                     f"= {100*PARAMS*2/itl/PEAK_BW:.0f}% of roof")
    for n in [2048, 4096, 8192]:
        ttft = prefill.ttft_ms[prefill.prompt_tokens == n].iloc[0] / 1e3
        flops = 2 * PARAMS * n + 2 * n * n * HIDDEN * LAYERS
        ax.scatter([flops / (2 * PARAMS)], [flops / ttft], s=90, c="tab:blue", zorder=5,
                   label=f"prefill n={n} ({100*flops/ttft/PEAK_FLOPS:.0f}% MFU)" if n == 2048 else None)
    ax.scatter([], [], s=90, c="tab:blue", label="prefill n=4096, 8192")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 6e3); ax.set_ylim(1e11, 8e14)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved throughput (FLOP/s or byte/s)")
    ax.set_title("Roofline — Qwen2.5-7B (bf16) on A100-SXM4-80GB")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "roofline.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved figures.png, roofline.png to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("loading engine...")
    llm = build_engine()
    kv = kv_capacity(llm)
    weights_gib = llm.llm_engine.model_config.hf_config.num_parameters * 2 / 2**30 \
        if hasattr(llm.llm_engine.model_config, "hf_config") else 14.25
    print(f"KV cache: {kv:,} tokens" if kv else "KV cache: unknown")
    print(f"concurrency at {MAX_LEN} tokens/request: {kv/MAX_LEN:.0f}x" if kv else "")

    print("\nprefill sweep:")
    prefill = sweep_prefill(llm)
    print("\ndecode sweep:")
    decode = sweep_decode(llm)
    print("\nconcurrency sweep:")
    throughput = sweep_concurrency(llm)

    prefill.to_csv(out / "prefill.csv", index=False)
    decode.to_csv(out / "decode.csv", index=False)
    throughput.to_csv(out / "throughput.csv", index=False)

    roofline_report(prefill, decode, throughput, weights_gib)
    plot_all(out, prefill, decode, throughput)


if __name__ == "__main__":
    main()
