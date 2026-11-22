# Dualv100AwesomeSetup

**The Complete Optimization Recipe for a Dual NVIDIA Tesla V100 (64GB HBM2) + HP Z8 G4 AI Workstation**

Turning passively-cooled datacenter **NVIDIA Tesla V100 PCIe 32GB (`sm_70` Volta)** cards inside a dual-socket **HP Z8 G4 Workstation** into a quiet, thermally rock-solid, high-throughput local LLM & Multimodal server.

---

## What Makes This Setup Awesome (Key Results)

| Metric | Stock / Unoptimized | With `Dualv100AwesomeSetup` | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Idle GPU Power & Temp** | ~45–55W/card, 55°C+ | **~28–30W/card (135 MHz Eco Mode), 42–47°C** | Automatic 120s idle clock-down (`v100-thermal-guard.py`) |
| **Sustained Load Temp** | 78°C–85°C+ (thermal throttling / loud fans) | **59°C–69°C** (+8°C to +18°C safety margin) | 150W–155W sweet-spot cap + **1380 MHz clock lock** + custom quiet duct |
| **111GB MoE Model (`UD-Q4_K_XL`)** | Heavy UPI cross-socket bottleneck / OOM | **51.0 tok/s single** (56.6 peak) / **61.9 tok/s parallel** | Socket-1 NUMA binding + 40 GiB RAM resident expert cache + Batch MTP |
| **GPU 1 Thermal Balance (`UD-Q4_K_XL` vs `IQ3_XXS`)** | GPU 1 hits **66–74°C** @ ~130W (`IQ3_XXS`) | **GPU 1 drops to 50–61°C @ ~82W (-48W cooler!)** | Hybrid VRAM+RAM tiering shifts GPU 1 duty cycle, keeping the hotter PCIe slot **10–14°C cooler** while delivering 4-bit quality |
| **64GB All-in-VRAM (`IQ3_XXS` / `IQ3_S`)** | ~52.8 tok/s @ 74°C | **63.6 tok/s** (68.8 peak) @ 61–65°C | `spec=3` MTP draft decoding + `int8` KV cache (up to 131k context) |
| **CUDA 13 Cutoff Bypass (`sm_70` Port)** | Dropped in CUDA 13 (`CUBLAS_STATUS_ARCH_MISMATCH`) | **Full 2026 Strata feature parity on `engine-cuda12`** | CUDA 12.9 + `-DSTRATA_EXPERIMENTAL_SM60=ON` + custom Volta `mma.m8n8k4` Tensor Core kernels |
| **Base System Idle (CPU + RAM)** | High dual-Xeon idle draw | **~24–28W CPU (both sockets) / ~45–55W wall** | `cpu-eco-profile.service` (`powersave` + `balance_performance` EPP) |

---

## Hardware Architecture Profile

* **Chassis:** HP Z8 G4 Workstation (1125W/1450W PSU)
* **CPUs:** 2× Intel Xeon Gold 6140 @ 2.30 GHz (36 Physical Cores / 72 Threads, Dual-Socket LGA3647, 2 NUMA Nodes)
* **Memory:** 64 GB ECC DDR4-2666 RDIMM (4× 16 GB, ~2.8–3.8W idle draw)
* **Compute GPUs:** 2× **NVIDIA Tesla V100 PCIe 32GB** (64 GB total HBM2, Volta `sm_70`, PCIe 3.0 x16, both physically wired to **CPU Socket 1 / NUMA Node 1**)
* **Display GPU:** 1× NVIDIA GeForce GT 1030 2GB GDDR5 (PCIe 3.0 x4 — strictly isolated from compute via `CUDA_VISIBLE_DEVICES=0,1` and V100 name filtering)

---

## The 7 Core Optimizations (The Recipe)

### 1. The Thermal Sweet Spot & Multi-Stage Hysteresis Watchdog (`daemons/v100-thermal-guard.py`)
Passively cooled Tesla V100 cards inside a desktop workstation will rapidly hit 80°C+ at their default 250W TDP.
* **Active Compute Baseline (Stage 0, < 73°C):** Caps power to **150W–155W** (configurable live via `/etc/v100-power-cap`) and locks graphics clocks to **`1380 MHz`** (`nvidia-smi -lgc 1380,1380`). This retains **98%+ of stock compute throughput** while cutting heat and power draw dramatically.
* **Dynamic Eco Idle Mode:** After 120 seconds of `< 5%` utilization, automatically steps the cards down to VBIOS minimum power (`100W` cap) and **`135 MHz`** core clock (~28W actual draw per card). Wakes up to `1380 MHz` within 2 seconds of load.
* **Multi-Stage Thermal Hysteresis & Hardware Cutoff:**
  * **Stage 1 (`>= 73°C`, recovers `<= 68°C`):** Steps down to `115W` cap & `1230 MHz`.
  * **Stage 2 (`>= 75°C`, recovers `<= 71°C`):** Steps down to `100W` cap & `1140 MHz`.
  * **Emergency Abort (`>= 80°C`):** Automatically sends `SIGTERM`/`SIGKILL` to compute PIDs on the overheating GPU to protect hardware.
* **Display GPU Isolation:** Strictly filters by `"V100"` in the GPU name so the GT 1030 display card is never touched.

### 2. Dual-Socket NUMA Affinity & UPI Interconnect Elimination (`systemd/strata-server.service`)
On the HP Z8 G4, the high-bandwidth PCIe x16 slots hosting both Tesla V100 cards sit on **NUMA Node 1 (CPU Socket 1)**.
* Running inference across all 36 cores forces expert weights and KV transfers across the Intel UPI socket interconnect, tanking throughput.
* **The Fix:** Pin the server strictly to Socket 1 with `numactl --cpunodebind=1 --preferred=1` and restrict worker threads to a single socket (`--pool-workers 16` or `17`).

### 3. Volta `sm_70` Kernel & Batch MTP Environment Tuning
Modern inference engines often optimize for Ampere/Hopper (`sm_80+`) by default. Setting these environment variables in [`systemd/strata-server.service`](systemd/strata-server.service) unlocks dedicated Volta Tensor Core & warp paths:
```ini
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="STRATA_SM70_TABLE=1"
Environment="STRATA_PF_WMMA=1"
Environment="STRATA_PF_WMMA2=1"
Environment="STRATA_HC_Q8=1"
Environment="STRATA_MTP_STAGE0=1"
Environment="STRATA_ARGMAX_ROWS_SM70=1"
Environment="STRATA_GR_NORM_FAST=1"
Environment="STRATA_GDN_REC_WARP=2"
Environment="STRATA_PREFILL_CHUNK=3072"
Environment="STRATA_CUDA_MALLOC_ASYNC=1"
Environment="STRATA_SM70_GEMM_SPLIT=1"
Environment="STRATA_BATCH_ATTN_ROWS=1"
Environment="STRATA_BATCH_GDN_GROUPS=1"
Environment="STRATA_BATCH_MTP_SERIAL=1"
Environment="STRATA_PW_WITH_BATCH=1"
Environment="STRATA_ADMIT_SHORT_FIRST=2048"
Environment="STRATA_MMQ_KQ=1"
```
* **Speculative Decoding (`--spec 3 --spec-min-p 0.38 --batch-mtp`):** Lowering MTP speculation depth from 4 to **3** reduced rejected draft waste, running **both cooler and faster** (67.1% draft acceptance rate across parallel streams!).
* **Per-Layer vs. Global Expert Cache:** Benchmarking proved that global `--expert-cache auto` with `--resident-budget-gib 40` and `--pipeline-windows 2 --lookup-chain 2` outperforms per-layer caching by **+15% tok/s** on MoE models.

### 4. Why 111GB `UD-Q4_K_XL` Actually Runs Cooler Than 47GB `IQ3_XXS` (The Thermal Paradox)
In a dual-slot workstation like the HP Z8 G4, **GPU 1 sits directly adjacent to GPU 0** and intakes warmer air, making **GPU 1 the thermal bottleneck** of the system.
* **With `IQ3_XXS` (47 GB, 100% in VRAM):** Every routed expert lives in HBM2 across both cards. During sustained generation, GPU 1 draws **~130W continuously** and reaches **66°C–74°C** (leaving only 4°C–12°C margin before the 78°C throttle threshold).
* **With Unsloth `UD-Q4_K_XL` (111 GB 4-bit, Hybrid VRAM + Pinned RAM):**
  * Strata places **14,985 hot experts (~43.8 GiB)** in V100 HBM2 across both cards (**93.2%–97.3% VRAM hit rate**) and pins **31.67 GiB** of the remaining experts in Socket-1 DDR4 RAM (`--resident-budget-gib 40`), with GPU 0 also hosting the shared MTP draft layer (`STRATA_MTP_STAGE0=1`).
  * Because stage hand-offs (`layers 0–21` on GPU 0 -> `layers 22–47` on GPU 1) pace GPU 1 alongside Socket-1 AVX-2/PCIe expert reads for the 3–7% cold tail, **GPU 1's average power draw drops from ~130W down to just 75W–85W** (~204W total dual-GPU draw vs. ~247W on `IQ3_XXS`).
  * **Result:** You upgrade from 3-bit (`IQ3_XXS`) to near-lossless 4-bit/5-bit dynamic quantization (`UD-Q4_K_XL`, matching `Q8_0` top tokens **93.1%** of the time), retain **51.0 tok/s single-stream / 61.9 tok/s parallel**, and **drop GPU 1 peak temperatures by 10°C–14°C (down to 50°C–61°C)**!

### 5. Protecting `/dev/shm` from Systemd SSH Logout Wipeout
By default on Ubuntu/Debian/Fedora, `systemd-logind` has `RemoveIPC=yes` and `Linger=no`. Disconnecting an SSH session wipes shared memory segments in `/dev/shm`, crashing persistent vision encoders and IPC caches.
```bash
sudo loginctl enable-linger $USER
sudo mkdir -p /etc/systemd/logind.conf.d
echo -e "[Login]\nRemoveIPC=no" | sudo tee /etc/systemd/logind.conf.d/no-remove-ipc.conf
sudo systemctl restart systemd-logind
```

### 6. Defeating the CUDA 13 Cutoff: The Volta `sm_70` Port (`engine-cuda12`)
When NVIDIA released **CUDA 13.0**, they hard-removed Volta (`sm_70`), Pascal (`sm_6x`), and Maxwell (`sm_5x`) from `nvcc`, `ptxas`, and `libcublas.so.13` (raising the hard floor to Turing `sm_75`).
* **Why PTX JIT workarounds fail on V100:** Community attempts to force CUDA 13 via PTX JIT (`-gencode arch=compute_70,code=compute_70` or patching `compute_75` headers) fail in production because Volta uses first-generation `mma.sync.aligned.m8n8k4` (`HMMA.884`) Tensor Core instructions (incompatible with Turing+ `mma.m16n8k8`), and CUDA 13's `cuBLAS` stripped all `sm_70` GEMM kernels (`CUBLAS_STATUS_ARCH_MISMATCH`).
* **The Real Fix (`-DSTRATA_EXPERIMENTAL_SM60=ON` on CUDA 12.8/12.9):** Rather than losing new 2026 inference features, Strata maintains a dedicated **`engine-cuda12` architecture port** built from the exact same latest source tree using **CUDA 12.8/12.9** paired with the modern **NVIDIA R580+ driver**:
  ```bash
  # Build the main Strata engine with full Volta sm_70 Tensor Core backports
  cmake -S engine -B build-cuda12 -G Ninja \
        -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_EXPERIMENTAL_SM60=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70" \
        -DCMAKE_BUILD_TYPE=Release
  ninja -C build-cuda12 strata

  # Build the multimodal vision encoder (strata-vision) for Volta sm_70
  cmake -B build-vision -S tools/vision -G Ninja \
        -DCMAKE_CUDA_ARCHITECTURES="70" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON
  ninja -C build-vision strata-vision
  ```
* **What `-DSTRATA_EXPERIMENTAL_SM60=ON` unlocks on V100:**
  * Lowers the CMake & runtime floor from `sm_75` to `sm_60`/`sm_70`.
  * Compiles `prompt_attn_v70_kernel` (native Volta `mma.m8n8k4` FP16 Tensor Core prompt attention, +18% prompt prefill speed).
  * Automatically converts `BF16` projections (hyper-connection, router, QSA indexer) to run on Volta's `FP16` Tensor Cores (`s884` cuBLAS/CUTLASS) instead of falling back to slow `FP32` SIMT kernels.
  * Reserve `576–960 MiB` VRAM on GPU 0 (`--vram-reserve-mib 960`) so the vision projector (`mmproj-BF16.gguf`) never collides with the `int8` KV cache.

### 7. Live Web Telemetry Dashboard & One-Click Power Cap Switcher (`daemons/hp-thermal-monitor.py`)
A zero-dependency Python web dashboard on port `9000` that monitors:
* Real-time GPU 0 & GPU 1 temperatures, HBM2 VRAM usage, clocks, and power draw.
* **Asymmetric GPU 1 Thermal Offset:** When switching power caps (`140W / 150W / 160W / 170W`) from the web UI, GPU 1 (which sits warmer in the chassis airflow) automatically receives a `-20W` offset to keep both cards at identical peak temperatures.
* Intel RAPL hardware energy telemetry (`/sys/class/powercap/intel-rapl:*`) for per-socket Xeon wattage.
* Live tok/s prefill, decode, prompt cache hit rate, and MTP draft acceptance parsed directly from the inference logs.

---

## Benchmark Progression Log

Full benchmark progression across cooling duct revisions, power caps, NUMA pinning, and Volta kernel flags:

| Date | Profile / Power Cap | Model | Avg Throughput | GPU 0 Peak | GPU 1 Peak | Avg Power (GPU 0 / 1) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Oct 25 (Baseline) | 160W Cap (Old Duct) | `qwen3.8-flash-next-iq3_xxs` | ~52.8 tok/s | 68.0°C | 74.0°C | ~140W total | Original duct, loud fan curve |
| Oct 25 | 150W Cap (New Quiet Duct) | `qwen3.8-flash-next-iq3_xxs` | 59.3 tok/s | 59.0°C | 64.0°C | ~112W total | Redesigned quiet duct (-10°C cooler!) |
| Oct 25 | 155W Cap, `spec=3` | `qwen3.8-flash-next-iq3_xxs` | 62.9 tok/s (68.1 peak) | 61.0°C | 66.0°C | 114.2W / 130.2W | Lowering MTP spec 4 -> 3 = faster & cooler |
| Oct 25 | 170W Cap, `spec=3` | `qwen3.8-flash-next-iq3_xxs` | **63.6 tok/s (68.8 peak)** | 61.0°C | 65.0°C | ~128W avg | Optimal all-in-VRAM 64GB profile |
| Oct 26 | 150W/120W, Volta-Opt | `qwen3.8-flash-next-ud-q4_k_xl` (111GB) | 43.1 tok/s (45.7 peak) | 59.0°C | 50.0°C | ~125W / ~75W | 111GB Unsloth Q4 + 40GB RAM budget + `SM70` flags |
| Oct 26 | 150W/150W, NUMA1 + 3072 chunk | `qwen3.8-flash-next-ud-q4_k_xl` (111GB) | 44.1 tok/s (46.9 peak) | 59.0°C | 52.0°C | ~130W / ~95W | Socket 1 NUMA pin + 3072 prefill chunking |
| Nov 05 | 150W/150W, `--adapt-async 1 --batch-mtp` | `qwen3.8-flash-next-ud-q4_k_xl` (111GB) | **51.0 tok/s (56.6 peak)** | 69.0°C | 61.0°C | 122.4W / 81.9W | Single-stream with async expert adaptation |
| Nov 05 | 150W/150W, **2× Concurrent Streams** | `qwen3.8-flash-next-ud-q4_k_xl` (111GB) | **61.9 tok/s agg (64.6 peak)** | 70.0°C | 61.0°C | ~130W / ~85W | 31.1 tok/s per stream, 67.1% MTP draft accept |

---

## Repository Structure

```text
Dualv100AwesomeSetup/
├── install.sh                         # One-step installer for OS tweaks, thermal guard & dashboard
├── daemons/
│   ├── v100-thermal-guard.py          # Dynamic Eco Idle (135MHz) + 1380MHz lock + 3-stage thermal guard
│   └── hp-thermal-monitor.py          # Port 9000 live web telemetry dashboard & power-cap switcher
├── systemd/
│   ├── cpu-eco-profile.service        # Dual Xeon balance_performance EPP & powersave governor
│   ├── v100-thermal-guard.service     # Systemd unit for the V100 thermal watchdog
│   ├── hp-thermal-dashboard.service   # Systemd unit for the web dashboard
│   └── strata-server.service          # NUMA Node 1 pinned Strata service with all 17 Volta SM70 env flags
├── configs/
│   ├── strata-ud-q4_k_xl.json         # 111GB Unsloth UD-Q4_K_XL config (64GB VRAM + 40GB RAM expert cache)
│   └── strata-iq3_xxs.json            # 131k-context all-in-VRAM IQ3_S config
├── scripts/
│   └── patch_server.py                # OpenAI SSE streaming finish_reason compatibility patch
├── benchmarks/
│   ├── test_parallel_bench.py         # Multi-stream concurrent benchmark with MTP acceptance metrics
│   ├── run_strata_load_test.py        # 5-round thermal stress test with 77°C emergency cutoff
│   ├── test_prompt_bench.py           # 4k-token prompt prefill benchmark
│   ├── gpu_test_dual.py               # Dual-V100 raw compute stress test
│   └── results/                       # Raw JSON telemetry from benchmark runs
└── docs/
    ├── STRATA_AND_OPENCLAW_GUIDE_SV.md # Full Swedish postmortem & step-by-step installation manual
    └── BENCHMARKS_SV.md                # Full Swedish benchmark log table
```

---

## Quick Start Installation

Clone the repository onto your HP Z8 G4 (or any Linux Dual-V100 workstation) and run the installer:

```bash
git clone https://github.com/mikasjoman/Dualv100AwesomeSetup.git
cd Dualv100AwesomeSetup
sudo ./install.sh
```

Then open `http://<your-workstation-ip>:9000` in your browser to view live GPU/CPU thermals, RAPL wattage, and token throughput.
