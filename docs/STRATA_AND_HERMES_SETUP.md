# Strata v0.2.19 & Hermes Agent: Dual-Node Batching & Subagent Setup Guide

This guide documents how to pair a **Dual NVIDIA Tesla V100 (64GB HBM2) LAN Server** running **Strata v0.2.19 (Engine `0.1.65`)** with a **Client Workstation GPU** (e.g., an RTX 30/40/50-series card running local Ollama) to power **Hermes Agent** (`NousResearch/hermes-agent`) with **2-slot parallel batching**, **prompt-cached subagents**, and **git worktree isolation**.

> **For AI Coding Agents:** You can follow the exact configuration blocks and commands in this file to reproduce this entire dual-node setup end-to-end.

---

## 1. Architecture Overview: Why Dual-Node Hybrid?

When running an autonomous coding agent like **Hermes** against a local MoE server, a single session performs two very different workloads:
1. **Heavy Frontier Reasoning & Parallel Subagents (Main Coding Loop):** Needs a massive model (`Qwen3.8-Flash-Next-UD-Q4_K_XL`, 111GB / 46.8GiB packed + resident experts, 98K context) with sticky KV prompt caching and parallel batch slots (`--batch 2`).
2. **High-Frequency Auxiliary Side-Tasks:** Context compression, title generation, command-safety approval, web extraction, memory flushing, and quick vision checks.

**The Single-Node Trap:** If Hermes sends its auxiliary side-tasks (like context compression or title generation) to the main V100 server, those side requests compete for batch slots and **evict or fragment the active coding conversation's KV prompt cache**.

**The Dual-Node Solution:**
* **Node A — LAN Compute Server (`192.168.1.240:8086`, Dual Tesla V100 64GB HBM2):** Dedicated **100%** to `model.default`, `delegation` (parallel subagents), `review`, and `goal_judge`. Runs with `--batch 2` (`"parallel": 2`), `--batch-mtp`, `--prompt-cache-spread`, and `--conversation-cache-rewind-keep 1`.
* **Node B — Local Workstation (`127.0.0.1:11434`, e.g., RTX 4070 SUPER 12GB running Ollama):** Handles all **10 auxiliary background pipelines** (`compression`, `vision`, `title_generation`, `approval`, `web_extract`, `session_search`, `kanban_diagnose`, `flush_memory`, `skill_review`, `wiki_ingest`) using fast local 7B–9B models (`qwen3.5:9b` at ~125 tok/s and `qwen2.5vl:7b`) with zero network latency and **zero V100 KV-cache disruption**.

---

## 2. LAN Server Setup: Strata v0.2.19 (Engine `0.1.65`) with Parallel Batching

### A. Compiling Strata v0.2.19+ for Volta `sm_70` (Dual V100)
Starting in Strata `0.2.16+`, the repository layout moved the C++ engine into `engine/` and the Python server into `terminal/serve/`. Always match the host C/C++ compiler (`gcc-13` / `g++-13`) to the `nvcc` CUDA 12 host compiler so `libstdc++` symbols link cleanly:

```bash
cd /home/user/strata
git pull origin main

# Ensure backward-compatible symlinks for top-level service paths
ln -sfn terminal/serve /home/user/strata/serve
ln -sfn terminal/data /home/user/strata/data
ln -sfn terminal/strata_control /home/user/strata/strata_control
ln -sfn terminal/strata_cli /home/user/strata/strata_cli

# Configure & build for Volta sm_70 (-DSTRATA_EXPERIMENTAL_SM60=ON unlocks sm_70 Tensor Cores)
cmake -S engine -B build-cuda12 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
      -DCMAKE_CUDA_ARCHITECTURES="70" \
      -DSTRATA_EXPERIMENTAL_SM60=ON \
      -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_GEOMETRY=flash_next \
      -DSTRATA_BUILD_TESTS=OFF

ninja -C build-cuda12 -j36 strata
cp -f build-cuda12/strata engine-cuda12/strata
```

### B. Enabling 2-Slot Parallel Batching & Sticky Prompt Caching (`configs/strata-ud-q4_k_xl.json`)
To allow Hermes to run **2 concurrent subagents** (or main agent + subagent) simultaneously without evicting each other's KV cache, configure `"parallel": 2` and the conversation/prompt cache flags in the Strata config JSON:

```json
{
  "exe": "/home/user/strata/engine-cuda12/strata",
  "args": [
    "--pack", "/home/user/Strata-data/packs/ud_q4_k_xl",
    "--native", "/home/user/Strata-data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--resident-budget-gib", "40",
    "--expert-profile", "/home/user/strata/data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "3072",
    "--spec", "3",
    "--spec-min-p", "0.38",
    "--mtp", "/home/user/Strata-data/mtp/rt",
    "--max-context", "98304",
    "--kv", "int8",
    "--remote-expert-opt",
    "--pipeline-windows", "2",
    "--lookup-chain", "2",
    "--vision",
    "--vram-reserve-mib", "512",
    "--pool-workers", "16",
    "--expert-profile-save", "/home/user/strata/data/expert-profile.bin",
    "--expert-profile-save-every", "5",
    "--layer-split", "auto",
    "--batch-mtp",
    "--adapt-async", "1",
    "--conversation-cache-min-tokens", "4096",
    "--conversation-cache-park-gain", "512",
    "--prompt-cache-spread",
    "--conversation-cache-rewind-keep", "1"
  ],
  "cwd": "/home/user/strata",
  "tokenizer": "/home/user/Strata-data/packs/ud_q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
  "log": "/home/user/strata/strata-ud-q4_k_xl.log",
  "port": 8086,
  "cuda": 12,
  "gpu": [0, 1],
  "gpus_asked": true,
  "layer_split": "auto",
  "host": "0.0.0.0",
  "open_browser": false,
  "fit_max_tokens": true,
  "reasoning_close_retry": true,
  "reasoning_budget_tokens": 4096,
  "tool_call_recovery": true,
  "parallel": 2
}
```

### C. Critical Batching & Volta Flags in `systemd/strata-server.service`
In addition to the standard Volta `sm_70` flags, Strata `v0.2.19` introduces three major dual-GPU layer-split flags that cut short/medium prompt prefill time in half:
* `STRATA_SPLIT_CPU_SHARE=1`: Co-schedules idle CPU worker cores to process streamed experts during 2-GPU layer-split prompt prefill (cuts 32–2,048 token prompt latency by **45%–55%**).
* `STRATA_MMQ_TIER_SORT=1`: Sorts prompt experts by readiness tier (HBM2 VRAM $\rightarrow$ Pinned RAM $\rightarrow$ File Stager) and hands hidden states between GPU 0 and GPU 1 via async P2P copies.
* `STRATA_POOL_DEP=1`: Enables the per-expert CPU pool dependency queue.

### D. Hardware BIOS Fan Floor (`25%` Minimum) for Passive V100s + NVMe Protection
In the HP Z8 G4 BIOS (*Advanced $\rightarrow$ Thermal*), set **Increase Idle Fan Speed(%)** to **`25%`** (up from `20%`).
* **Why:** The onboard M.2 NVMe PCIe controller sits directly in the PCIe airflow zone. At 20% fan speed during 40 GiB model loads, the bare NVMe ASIC controller (`Sensor 2`) can spike to `98.8°C` (thermal throttling). Bumping the BIOS idle fan floor to **25%** crosses the chassis static-pressure threshold, dropping the NVMe controller by **`-47.9°C`** (down to `50.9°C`) and allowing both V100s to cool down to **`38°C`** in Eco Idle mode.

---

## 3. Hermes Agent Setup: Subagents, Batching & Dual-Node Routing

Create or update `~/.hermes/config.yaml` on your client workstation with the following production configuration:

```yaml
model:
  default: qwen3.8-flash-next-ud-q4_k_xl
  provider: strata
  base_url: http://192.168.1.240:8086/v1
  supports_vision: true

fallback_providers:
  - provider: strata-local

# 1. Dual-Node Provider Definitions
providers:
  # Primary Frontier Engine: LAN Dual Tesla V100 Server (64GB HBM2)
  strata:
    base_url: http://192.168.1.240:8086/v1
    api_key: sk-strata
    context_length: 98304
    extra_body:
      max_tokens: 16384
      reasoning_budget_tokens: 4096

  # Local Auxiliary Text Engine: Client GPU running Ollama (e.g., RTX 4070 SUPER 12GB)
  strata-local:
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    context_length: 32768
    default: qwen3.5:9b

  # Local Auxiliary Vision Engine: Client GPU running Ollama
  local-vision:
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
    context_length: 32768
    default: qwen2.5vl:7b

# 2. Parallel Subagent & Batching Configuration (Matched 1:1 with Strata --batch 2)
delegation:
  provider: strata
  model: qwen3.8-flash-next-ud-q4_k_xl
  reasoning_effort: medium
  max_concurrent_children: 2        # Matches Strata's "parallel": 2 (--batch 2) slots
  oneshot_max_children: 4
  max_iterations: 250
  prefix_stagger_seconds: 8.0       # Child #2 waits for Child #1's first token so both share 100% of the prompt cache!
  worktree_isolation: true          # Each parallel subagent gets its own isolated git worktree off HEAD
  subagent_auto_approve: true       # Prevents background worker threads from silently auto-denying build/test commands
  independent_completions: true     # Streams each subagent's result back to the parent as soon as it finishes
  max_spawn_depth: 2                # Enables Orchestrator (depth 1) -> Leaf Worker (depth 2) hierarchies
  orchestrator_enabled: true
  request_overrides:
    max_tokens: 16384
    reasoning_budget_tokens: 4096

# 3. Auxiliary Offload to Local Workstation GPU (Protects V100 KV Prompt Cache!)
auxiliary:
  vision:
    provider: local-vision
    model: qwen2.5vl:7b
    reasoning_effort: medium
    timeout: 120
  compression:
    provider: strata-local
    model: qwen3.5:9b
    reasoning_effort: medium
    timeout: 180
  title_generation:
    enabled: true
    model_upgrade_enabled: false
    provider: strata-local
    model: qwen3.5:9b
    reasoning_effort: medium
    timeout: 60
  approval:
    provider: strata-local
    model: qwen3.5:9b
    reasoning_effort: medium
    timeout: 60
  web_extract:
    provider: strata-local
    model: qwen3.5:9b
  session_search:
    provider: strata-local
    model: qwen3.5:9b
  kanban_diagnose:
    provider: strata-local
    model: qwen3.5:9b
  flush_memory:
    provider: strata-local
    model: qwen3.5:9b
  skill_review:
    provider: strata-local
    model: qwen3.5:9b
  wiki_ingest:
    provider: strata-local
    model: qwen3.5:9b
  # Heavy code review & goal verification stay on the 111GB V100 model
  review:
    provider: strata
    model: qwen3.8-flash-next-ud-q4_k_xl
    reasoning_effort: medium
  goal_judge:
    provider: strata
    model: qwen3.8-flash-next-ud-q4_k_xl
    reasoning_effort: medium
    timeout: 90

# 4. Context Compression Hygiene
compression:
  enabled: true
  threshold: 0.82
  target_ratio: 0.15
  protect_first_n: 2
  protect_last_n: 4
  proactive_prune_tokens: 1024
```

---

## 4. How the Subagent & Batching Mechanics Work Together

When an agent (or user) configures this stack, six mechanisms interlock between **Hermes** and **Strata**:

1. **1:1 Batch Slot Alignment (`max_concurrent_children: 2` $\leftrightarrow$ `"parallel": 2` / `--batch 2`):**
   Strata allocates 2 hardware KV-cache slots across the Dual V100s (`--batch 2 --batch-mtp`). Setting `max_concurrent_children: 2` ensures Hermes never spawns a 3rd concurrent heavy subagent that would queue behind or evict an active slot.
2. **Prompt-Cache Warm-Up Gate (`prefix_stagger_seconds: 8.0` $\leftrightarrow$ `--prompt-cache-spread`):**
   Sibling subagents in a batch share an identical ~15,000-token prefix (system prompt + tool schemas). Hermes's `PrefixCacheGate` launches Subagent #1 immediately and holds Subagent #2 for a few hundred milliseconds until Subagent #1 starts streaming its first token. Subagent #2 then gets a **100% prompt-cache hit** in Strata's VRAM instead of paying a second 15,000-token prefill!
3. **Zero-Conflict Parallel File Edits (`worktree_isolation: true`):**
   Each parallel subagent automatically runs inside its own temporary `git worktree` branched from the parent's `HEAD`. Two subagents can edit code and run compilers/tests simultaneously without file race conditions or lock contention.
4. **Non-Blocking Background Execution (`subagent_auto_approve: true` & `independent_completions: true`):**
   Worker threads cannot prompt an interactive CLI for command approval (and default to auto-denying unknown commands). `subagent_auto_approve: true` allows workers to run test/build suites autonomously, and `independent_completions: true` delivers each worker's result to the parent agent the moment it finishes.
5. **Preventing Output Cutoffs (`max_tokens: 16384` + `reasoning_budget_tokens: 4096`):**
   When a thinking model generates large code blocks, Strata caps thinking tokens to leave room for the final answer. Setting `max_tokens: 16384` with `reasoning_budget_tokens: 4096` guarantees up to **12,288 tokens of pure code/tool output** after thinking completes, eliminating `"The model's reply was cut off before it finished"` errors.
6. **Skill Index & Toolset Hygiene (`hermes prompt-size`):**
   Always run `hermes prompt-size` after installing community skill packs. Keep active skills scoped to the current domain (ideally `< 70` skills / `< 6 KB` index size) by listing unused skills under `skills.disabled:` in `~/.hermes/config.yaml`, and disable unused heavy toolsets (`browser`, `tts`, `image_gen`, `computer_use`) under `disabled_toolsets:`.
