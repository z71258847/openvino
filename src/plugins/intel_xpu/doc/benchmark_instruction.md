# XPU Hybrid Benchmark — Reproduction Guide (clean machine)

This document is a complete, step‑by‑step recipe to reproduce the **llama‑3.1‑8B INT4
channel‑wise, 1024‑in / 128‑out** prefill+decode benchmark across three configurations on a
fresh Intel Core Ultra machine:

| Config | Device | Runtime stack | What it measures |
|--------|--------|---------------|------------------|
| **GPU** | iGPU | stock OpenVINO **GenAI** (pip) | baseline: GenAI `LLMPipeline` prefill/decode |
| **NPU** | NPU  | stock OpenVINO **GenAI** (pip) | baseline: GenAI static NPUW prefill/decode |
| **XPU** | GPU+NPU | **custom OpenVINO build** (this fork) | hybrid: GPU prefill + NPU decode, shared INT4 weights + KV |

> The XPU hybrid is a custom meta‑plugin in this fork (`src/plugins/intel_xpu`). It is **not**
> in stock GenAI, so it is driven by the raw‑runtime harness `bench_npuw_vs_xpu.py`. That harness's
> prefill sub‑model emits **last‑token logits only** (`[1,1,V]`), which is methodologically
> equivalent to how GenAI runs GPU/NPU — so the three numbers are comparable.

---

## 0. Hardware / OS prerequisites

- Intel **Core Ultra** CPU with an NPU (Meteor Lake / Lunar Lake / **Panther Lake** — this guide
  was validated on **PTL**, device `PCI\VEN_8086&DEV_B03E`, "Intel AI Boost").
- **Windows 11** (23H2 / 24H2 / 25H2). NPU workloads only run on Windows 11.
- ~60 GB free disk (source + build + model), ≥ 32 GB RAM recommended.
- Integrated Intel GPU driver installed (usually present; update from the OEM / Intel if not).

---

## 1. Install the Intel NPU driver

The benchmark needs a working NPU **runtime + firmware**. The NPU **compiler (VCL)** is *not* taken
from the driver — it ships inside the custom OpenVINO build (see §5.4) — so the ordinary **public**
driver is sufficient. You do **not** need the internal test‑signed package for this benchmark.


1. Download **Intel® NPU Driver – Windows** from the Intel Download Center:
   <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>
   (validated build: `32.0.100.4778`; newer `32.0.100.xxxx` is fine).
2. Run the installer `.exe` as Administrator, follow the prompts, reboot if asked.
   - No test‑signing, no certificate import, no Secure Boot changes are required — it is
     WHQL/attestation‑signed.
3. Verify: Device Manager → **Neural processors → Intel AI Boost** shows a `32.0.100.xxxx`
   driver with no yellow bang.


---

## 2. Install the build toolchain

- **Visual Studio 2022** (Community is fine) with the **"Desktop development with C++"** workload
  (MSVC v143, Windows 11 SDK).
- **CMake** ≥ 3.24 (the VS bundled one works; or install standalone and add to PATH).
- **Git** (with support for submodules).
- **Python 3.11** (validated: 3.11.3) on PATH as `python`. (3.10–3.12 also work.)
- **Ninja** (optional, faster than the VS generator).

---

## 3. Get the source

Clone this fork (or copy the whole `openvino/` tree) and pull submodules:

```bat
git clone https://github.com/z71258847/openvino C:\ov\openvino
cd C:\ov\openvino
git checkout yq/xpu-shared-buffer
git submodule update --init --recursive
```

The XPU meta‑plugin lives at `src/plugins/intel_xpu` and is auto‑discovered by the plugin build
(`src/plugins/CMakeLists.txt` iterates subdirectories) — **no special enable flag is needed**. It
does depend on the GPU and NPU plugins being built (they are the fused back‑ends).

---

## 4. Build the custom OpenVINO (with the XPU plugin + Python)

Install the Python build requirements first:

```bat
python -m pip install -r src\bindings\python\requirements.txt
python -m pip install -r src\bindings\python\wheel\requirements-dev.txt
```

Configure and build (Release). These flags match the validated build:

```bat
:: from an "x64 Native Tools Command Prompt for VS 2022", or after calling vcvars64.bat
cd C:\ov\openvino
cmake -B build -G "Visual Studio 17 2022" \
      -D CMAKE_BUILD_TYPE=Release \
      -D ENABLE_INTEL_GPU=ON  -D GPU_RT_TYPE=L0 \
      -D ENABLE_INTEL_NPU=ON \
      -D ENABLE_INTEL_CPU=ON \
      -D ENABLE_ONEDNN_FOR_GPU=ON \
      -D ENABLE_PYTHON=ON  -D Python3_EXECUTABLE=<path-to-python> \
      -D THREADING=TBB_ADAPTIVE
cmake --build build --config Release --parallel
```

**Verify the XPU plugin was produced and registered:**

```bat
type bin\intel64\Release\plugins.xml
```
must contain the three plugins:
```xml
<plugin name="GPU" location="openvino_intel_gpu_plugin.dll"></plugin>
<plugin name="NPU" location="openvino_intel_npu_plugin.dll"></plugin>
<plugin name="XPU" location="openvino_intel_xpu_plugin.dll"></plugin>
```

From here on, `RELEASE` refers to the custom build's binary dir:
```
C:\ov\openvino\bin\intel64\Release
```

---

## 5. Python environments

Two stacks are used **on purpose** and must stay separate:

### 5.1 Stock GenAI stack (for the GPU & NPU baselines) — pip

```bat
python -m pip install "openvino==2026.1.*" "openvino-genai==2026.1.*"
```
- The GPU/NPU baselines run against this **matched** pip pair (validated: ov `2026.1.0-21367`,
  genai `2026.1.0.0-2957`). Do **not** put the custom `RELEASE` dir on `sys.path` for these runs —
  keep the ABI matched.

### 5.2 Custom‑build stack (for the XPU hybrid) — no pip openvino

The harness loads the custom build explicitly (it does not rely on pip openvino):
```python
os.add_dll_directory(RELEASE)
sys.path.insert(0, os.path.join(RELEASE, "python"))   # avoids the namespace-stub "no Core" trap
```
This is already done inside `bench_npuw_vs_xpu.py`.

### 5.3 Model‑export stack (once, §6)

```bat
python -m pip install "optimum-intel[openvino]" nncf transformers
```

### 5.4 CLIntercept gotcha (affects GPU timing — important)

Some machines have the **CLIntercept** shim installed as `C:\Python311\OpenCL.dll` (it prints
`CLIntercept (64-bit) is loading...`). It instruments every OpenCL call and **inflates GPU time**.
Both harnesses preload a real OpenCL ICD loader first so the GPU plugin binds the clean loader:
```python
ctypes.CDLL(r"C:\Windows\System32\OpenCL.dll")   # or RELEASE\OpenCL.dll
```
If you see the `CLIntercept ... loading` banner in the GPU run output, the preload did **not** take
effect — fix the path before trusting the GPU number.

### 5.5 NPU compiler (VCL) note

With `NPU_COMPILER_TYPE=PLUGIN` (used by the NPU sub‑compile), OpenVINO loads its **bundled**
compiler from `RELEASE\openvino_intel_npu_compiler.dll` (validated `2026.2.0.1`), **not** the
driver's copy. That is why the public driver (§1a) is sufficient. To confirm which VCL is loaded on
a machine, use `vcl_probe.py` (compiles a tiny NPU model and dumps the mapped `*npu_compiler*.dll`).

---

## 6. Prepare the model (llama‑3.1‑8B INT4 channel‑wise)

`meta-llama/Llama-3.1-8B` is a gated HF model — accept the license and set a token first. On the
Intel corporate network you also need the DMZ proxy:

```bat
set HF_TOKEN=hf_xxxxxxxx
```

Export to INT4 **channel‑wise, symmetric** (cw = per‑channel = `--group-size -1`), all layers INT4:

```bat
optimum-cli export openvino ^
  -m meta-llama/Llama-3.1-8B ^
  --weight-format int4 --sym --group-size -1 --ratio 1.0 ^
  C:\ov\models\llama-3.1-8b-int4-cw-ov
```

Result (~4.3 GB `openvino_model.bin`): a directory with `openvino_model.xml/.bin`,
`openvino_tokenizer.*`, `openvino_detokenizer.*`, and the tokenizer/config JSONs. This is the
`MODEL` directory used below.

---

## 7. Copy the benchmark scripts and update paths

Two scripts drive everything (both committed in this doc dir alongside this file):

- `bench_npuw_vs_xpu.py` — raw‑runtime harness (modes `xpu`, `npuw`, `gpu`) → **XPU hybrid** number.
- `genai_bench.py` — stock **GenAI** harness (device `GPU` / `NPU`) → **baseline** numbers.

Copy both to a working dir on the target machine, then edit the machine‑specific paths:

**`bench_npuw_vs_xpu.py`**
- Line ~15: `RELEASE = r"...\bin\intel64\Release"` → your custom build's Release dir.
- `-m/--model` default (or pass `-m` on the CLI) → your exported model `.xml`.

**`genai_bench.py`**
- `MODEL = r"...\llama-3.1-8b-int4-cw-ov"` → your exported model **directory**.
- The `for _ocl in (...)` preload list — make sure one entry is a real OpenCL ICD on this machine
  (`C:\Windows\System32\OpenCL.dll` is the safe default).
- It uses the **pip** genai stack, so it intentionally does **not** import from `RELEASE`.

---

## 8. Run the benchmarks

Run **one device per process** (avoids OOM and device contention) and **cool down ~60 s between
runs** so GPU‑prefill heat doesn't bias a following NPU/XPU decode on the shared package.

```bat
:: --- stock GenAI baselines (pip stack) ---
python genai_bench.py GPU 1024 128 3
timeout /t 60
python genai_bench.py NPU 1024 128 3
timeout /t 60

:: --- XPU hybrid (custom build) ---
python bench_npuw_vs_xpu.py xpu 1024 128 3 -m C:\ov\models\llama-3.1-8b-int4-cw-ov\openvino_model.xml
```

Args are `<device/mode> <prompt_len> <max_new> <niter>`.

Optional cross‑checks (raw‑runtime GPU/NPU, i.e. *not* GenAI — note raw GPU emits full‑sequence
logits and will over‑report prefill; see §10):
```bat
python bench_npuw_vs_xpu.py npuw 1024 128 3 -m ...\openvino_model.xml
python bench_npuw_vs_xpu.py gpu  1024 128 3 -m ...\openvino_model.xml
```

To prove no content‑keyed caching, add `--unique-prompt` (fresh random prompt each iter) to the
`bench_npuw_vs_xpu.py` runs — prefill should be unchanged.

---

## 9. Reading the output

- **`genai_bench.py`** prints `TTFT` (time‑to‑first‑token = **prefill**) and `TPOT`
  (time‑per‑output‑token = **decode**) plus throughput, greedy, `ignore_eos` (full 128 tokens),
  prefix caching **off**.
- **`bench_npuw_vs_xpu.py`** prints `prefill … ms` and `decode steady … ms/tok`.

### Reference results (validated, PTL — expect device‑specific variation)

| Config | Prefill | Decode | Decode tput |
|--------|--------:|-------:|------------:|
| **GPU** (GenAI) | ~223 ms | ~39.0 ms/tok | ~25.6 tok/s |
| **NPU** (GenAI) | ~1165 ms | ~56.3 ms/tok | ~17.8 tok/s |
| **XPU** (hybrid) | ~295 ms | ~55.7 ms/tok | ~18.0 tok/s |

Interpretation: **GPU is fastest** on both phases. The hybrid's value is delivering **GPU‑class
prefill with NPU‑offloaded decode** — its real competitor is **pure‑NPU**, whose prefill it cuts
**~4×** (1165 → 295 ms) at equal decode. Against pure‑GPU the hybrid is a power/offload play, not a
speed play.

---

## 10. Methodology notes & pitfalls (read if numbers look off)

1. **Full‑sequence logits pitfall (GPU).** Driving the stateful model with raw `infer()`
   (`bench_npuw_vs_xpu.py gpu`) returns `[1, 1024, 128256]` fp32 = **525 MB** of logits — it runs
   the LM head over all 1024 positions and DMAs the whole thing back. That inflates GPU prefill to
   ~400 ms. **GenAI slices to the last token** (`[1,1,V]`, 0.5 MB) → the true ~223 ms. Always use
   `genai_bench.py` for the GPU/NPU baselines. The XPU hybrid already emits last‑token logits, so
   it is comparable.
2. **Prefix caching off.** `genai_bench.py` sets `SchedulerConfig.enable_prefix_caching = False`.
   The raw path has no prefix cache at all (that's a GenAI‑pipeline feature). Confirm with
   `--unique-prompt`.
3. **CLIntercept.** See §5.4 — bypass it or the GPU number is wrong.
4. **Cooldown.** The GPU and NPU share one package power/thermal budget; a hot GPU prefill throttles
   a subsequent NPU decode. Insert ~60 s idle between device runs (already scripted above).
5. **One process per device.** An 8B INT4 model on two devices will OOM if co‑resident; never run
   two modes in one process.
6. **XPU env flag.** The hybrid GPU prefill must read the shared INT4 weights with no reorder:
   `OV_XPU_REF_COMPRESSED_FC=1` (set automatically by `bench_npuw_vs_xpu.py` in `xpu` mode).
7. **Harmless XPU teardown crash.** The XPU meta‑plugin may emit a `Segmentation fault` at Python
   process exit (teardown AV). It happens **after** all results are printed — ignore it.
8. **KV sizing.** For prompts > 1024, the harness forwards `NPUW_LLM_MAX_PROMPT_LEN` /
   `MIN_RESPONSE_LEN`; the defaults (1024/128) overflow on longer prompts, so pass a larger
   `prompt_len`/`max_new` and the harness resizes the static KV accordingly.

---

## 11. Version stamps (validated configuration)

| Component | Version |
|-----------|---------|
| NPU driver (public) | `32.0.100.4778` |
| Bundled VCL (in custom build) | `2026.2.0.1` |
| Custom OpenVINO build | `2026.2.0.1` (VS2022, Release, GPU+NPU+CPU+Python) |
| pip openvino (GenAI baselines) | `2026.1.0-21367` |
| pip openvino‑genai | `2026.1.0.0-2957` |
| Python | `3.11.3` |
| Model | `meta-llama/Llama-3.1-8B`, INT4 sym channel‑wise (`--group-size -1`) |
| Device | PTL, `PCI\VEN_8086&DEV_B03E`, "Intel AI Boost" |
