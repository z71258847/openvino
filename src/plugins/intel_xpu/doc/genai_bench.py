r"""Stock OpenVINO GenAI baseline: prefill (TTFT) + decode (TPOT) for GPU / NPU.
Uses the pip-installed openvino_genai (matched openvino runtime) -- NOT the custom 2026.2 XPU build --
so this is a clean 'stock GenAI' number. Fixed 1024-token prompt, 128 new tokens, greedy, ignore_eos,
prefix-caching OFF. One device per process (avoid OOM / device contention).

Usage: genai_bench.py <GPU|NPU> [prompt_len] [max_new] [niter]
"""
import sys, ctypes, numpy as np
# Bypass the CLIntercept shim at C:\Python311\OpenCL.dll by preloading a REAL OpenCL ICD loader first,
# so the GPU plugin binds the clean loader (no per-call interception overhead) -> fair GPU timing.
for _ocl in (r"C:\Windows\System32\OpenCL.dll",
             r"C:\yqiu\xpu-buffer-ov\openvino\bin\intel64\Release\OpenCL.dll"):
    try: ctypes.CDLL(_ocl); break
    except OSError: pass
import openvino as ov
import openvino_genai as ov_genai

DEV     = sys.argv[1] if len(sys.argv) > 1 else "GPU"
L       = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 128
NITER   = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MODEL   = r"C:\yqiu\xpu-buffer-ov\llama-3.1-8b-int4-cw-ov"
print(f"ov={ov.get_version()}  genai={ov_genai.__version__}  device={DEV}", flush=True)

cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = MAX_NEW
cfg.min_new_tokens = MAX_NEW          # force full 128 tokens
cfg.ignore_eos     = True
cfg.do_sample      = False            # greedy
cfg.apply_chat_template = False

if DEV == "NPU":
    pipe = ov_genai.LLMPipeline(MODEL, "NPU", MAX_PROMPT_LEN=L, MIN_RESPONSE_LEN=max(MAX_NEW, 128))
else:
    sched = ov_genai.SchedulerConfig()
    sched.enable_prefix_caching = False           # <-- prefix caching disabled
    sched.max_num_batched_tokens = sys.maxsize
    pipe = ov_genai.LLMPipeline(MODEL, DEV, scheduler_config=sched)

# exact L-token prompt (valid llama3.1 ids), bypass tokenizer so prompt size is deterministic
ids  = np.array([[(i % 50000) + 100 for i in range(L)]], dtype=np.int64)
mask = np.ones((1, L), dtype=np.int64)
inp  = ov_genai.TokenizedInputs(ov.Tensor(ids), ov.Tensor(mask))

print(f"warmup (1)... prompt={L} tok, max_new={MAX_NEW}", flush=True)
pipe.generate(inp, cfg)

ttft, tpot, thr, gen = [], [], [], []
for i in range(NITER):
    res = pipe.generate(inp, cfg)
    pm = res.perf_metrics
    ttft.append(pm.get_ttft().mean)
    tpot.append(pm.get_tpot().mean)
    thr.append(pm.get_throughput().mean)
    gen.append(pm.get_num_generated_tokens())
    print(f"  iter{i}: gen={gen[-1]:3d}  TTFT(prefill)={ttft[-1]:7.1f} ms  "
          f"TPOT(decode)={tpot[-1]:6.2f} ms/tok  thr={thr[-1]:5.1f} tok/s", flush=True)

import statistics as st
print(f"\n[{DEV}] prompt={L} new={MAX_NEW}  "
      f"prefill(TTFT) {st.mean(ttft):.1f} ms   decode(TPOT) {st.mean(tpot):.2f} ms/tok "
      f"({1000/st.mean(tpot):.1f} tok/s)   over {NITER} iters (gen {gen})", flush=True)
