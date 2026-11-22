#!/usr/bin/env python3
"""
v100AIServerMonitor — Zero-Dependency Live Web Telemetry Dashboard & Power Controller
for Dual NVIDIA Tesla V100 AI Servers & Dual-Socket Xeon Workstations.

Features:
- Real-time GPU telemetry via `nvidia-smi` (temperatures, wattage, clocks, HBM2 VRAM).
- Live one-click Peak Power Cap switcher (140W / 150W / 160W / 170W) with optional
  thermal offset for secondary GPUs in tight workstation airflow.
- Intel RAPL hardware power telemetry (per-socket CPU wattage & ECC DRAM wattage).
- Live LLM inference engine monitoring (OpenAI-compatible `/v1/models` endpoint +
  log parser for prefill tok/s, decode tok/s, prompt cache hits, and MTP draft acceptance).
- Auto-discovery of `.gguf` and tensor files stored on local storage.
- Zero external Python dependencies (runs on standard library Python 3.8+).
"""

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ==============================================================================
# Configuration (Customizable via Environment Variables or CLI Arguments)
# ==============================================================================
PORT = int(os.environ.get("V100_MONITOR_PORT", "9000"))
BIND_HOST = os.environ.get("V100_MONITOR_HOST", "0.0.0.0")
SERVER_TITLE = os.environ.get("V100_MONITOR_TITLE", "Dual Tesla V100 AI Server Monitor")
SERVER_SUBTITLE = os.environ.get(
    "V100_MONITOR_SUBTITLE",
    "Dual NVIDIA Tesla V100 PCIe (2x 32GB = 64GB HBM2) & Dual Intel Xeon Workstation",
)

# Power Cap Management (shared with v100-thermal-guard if installed)
POWER_CAP_FILE = os.environ.get("V100_POWER_CAP_FILE", "/etc/v100-power-cap")
POWER_CAP_OPTIONS = [
    int(x) for x in os.environ.get("V100_POWER_CAP_OPTIONS", "140,150,160,170").split(",")
]
DEFAULT_POWER_CAP = int(os.environ.get("V100_DEFAULT_POWER_CAP", "150"))
# Optional wattage offset for GPU 1 (set to 20 if GPU 1 runs warmer in chassis airflow, or 0 for identical caps)
GPU1_POWER_OFFSET_W = int(os.environ.get("V100_GPU1_OFFSET_W", "20"))
BASE_SYSTEM_WATTS = float(os.environ.get("V100_BASE_SYSTEM_WATTS", "45.0"))

# AI Inference Server Integration (Strata / llama.cpp / vLLM OpenAI-compatible API)
INFERENCE_HOST = os.environ.get("V100_INFERENCE_HOST", "127.0.0.1")
INFERENCE_PORT = int(os.environ.get("V100_INFERENCE_PORT", "8086"))

# Comma-separated glob paths for inference logs and model directories
INFERENCE_LOG_GLOBS = os.environ.get(
    "V100_INFERENCE_LOGS",
    "/home/*/strata/*.log,/var/log/strata*.log",
).split(",")

MODEL_STORAGE_DIRS = os.environ.get(
    "V100_MODEL_DIRS",
    "/home/*/Strata-data/models,/home/*/Strata-data/models/*,/home/*/models",
).split(",")

ENABLE_SHUTDOWN_BUTTON = os.environ.get("V100_ENABLE_SHUTDOWN", "1").lower() in ("1", "true", "yes")


# ==============================================================================
# Hardware & Power Cap Helpers
# ==============================================================================
def get_active_power_cap() -> int:
    try:
        with open(POWER_CAP_FILE, "r", encoding="utf-8") as f:
            return int(float(f.read().strip()))
    except Exception:
        return DEFAULT_POWER_CAP


def apply_power_cap(watts: int):
    """Writes the target baseline power cap to POWER_CAP_FILE and applies it via nvidia-smi."""
    base_watts = int(watts)
    try:
        with open(POWER_CAP_FILE, "w", encoding="utf-8") as f:
            f.write(f"{base_watts}\n")
    except PermissionError:
        pass

    applied = []
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and "V100" in parts[1].upper():
                idx_int = int(parts[0])
                target_w = (
                    max(100, base_watts - GPU1_POWER_OFFSET_W)
                    if idx_int == 1
                    else base_watts
                )
                r = subprocess.run(
                    ["nvidia-smi", "-i", parts[0], "-pl", str(target_w)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if r.returncode == 0:
                    applied.append(f"GPU {parts[0]}:{target_w}W")
    except Exception:
        pass
    return base_watts, applied


def read_rapl_watts() -> dict:
    """Samples Intel RAPL energy counters over a 100ms window to compute live CPU & DRAM wattage."""
    rapl_paths = glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
    names = {}
    for p in rapl_paths:
        name_path = os.path.join(os.path.dirname(p), "name")
        try:
            with open(name_path, "r", encoding="utf-8") as f:
                names[p] = f.read().strip()
        except Exception:
            names[p] = os.path.basename(os.path.dirname(p))

    e1 = {}
    for p in rapl_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                e1[p] = int(f.read().strip())
        except Exception:
            pass

    if not e1:
        return {"Total CPU Watt": 0.0}

    time.sleep(0.1)

    e2 = {}
    for p in rapl_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                e2[p] = int(f.read().strip())
        except Exception:
            pass

    results = {}
    total_cpu = 0.0
    for p, val1 in e1.items():
        if p in e2:
            diff_uj = e2[p] - val1
            if diff_uj < 0:
                continue
            watts = (diff_uj / 1e6) / 0.1
            nm = names.get(p, "unknown")
            results[nm] = round(watts, 2)
            if "package" in nm:
                total_cpu += watts

    results["Total CPU Watt"] = round(total_cpu, 2)
    return results


# ==============================================================================
# Inference Performance & System Telemetry
# ==============================================================================
_last_known_inference_perf = {
    "prefill_tok_s": 0.0,
    "batch_prefill_tok_s": 0.0,
    "decode_tok_s": 0.0,
    "prompt_tokens": 0,
    "reused_tokens": 0,
    "read_tokens": 0,
    "prompt_cache_hit_pct": 0.0,
    "generated_tokens": 0,
    "drafts_accepted": "spec=3 (Optimal)",
    "expert_cache_hit": 100.0,
    "is_active": False,
    "has_data": False,
}


def get_inference_perf() -> dict:
    """Parses the most recently modified inference log for live prefill/decode tok/s and cache hit rates."""
    global _last_known_inference_perf
    candidate_logs = []
    for pattern in INFERENCE_LOG_GLOBS:
        candidate_logs.extend(glob.glob(pattern.strip()))

    log_path = None
    latest_mtime = 0.0
    for cand in candidate_logs:
        if os.path.isfile(cand):
            mt = os.path.getmtime(cand)
            if mt > latest_mtime:
                latest_mtime = mt
                log_path = cand

    if not log_path:
        return _last_known_inference_perf.copy()

    perf = _last_known_inference_perf.copy()
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-300:]

        found_turn = False
        for line in reversed(lines):
            if "prompt" in line and "read in" in line:
                m_details = re.search(
                    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms "
                    r"\(([\d\.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d\.]+) tok/s\)",
                    line,
                )
                if m_details:
                    perf["prompt_tokens"] = int(m_details.group(1))
                    perf["reused_tokens"] = int(m_details.group(2))
                    perf["read_tokens"] = int(m_details.group(3))
                    perf["prefill_tok_s"] = float(m_details.group(5))
                    perf["generated_tokens"] = int(m_details.group(6))
                    perf["decode_tok_s"] = float(m_details.group(8))
                    m_drafts = re.search(r"drafts accepted (\d+ of \d+)", line)
                    if m_drafts:
                        perf["drafts_accepted"] = m_drafts.group(1)
                    if perf["prompt_tokens"] > 0:
                        perf["prompt_cache_hit_pct"] = round(
                            (perf["reused_tokens"] / perf["prompt_tokens"]) * 100, 1
                        )
                    perf["has_data"] = True
                    found_turn = True
                    break
                else:
                    m_prefill = re.search(r"read in \d+ ms \(([\d\.]+) tok/s\)", line)
                    m_gen = re.search(r"generated in \d+ ms \(([\d\.]+) tok/s\)", line)
                    m_tokens = re.search(r"prompt (\d+) tokens.*?, (\d+) generated", line)
                    m_drafts = re.search(r"drafts accepted (\d+ of \d+)", line)
                    if m_prefill:
                        perf["prefill_tok_s"] = float(m_prefill.group(1))
                    if m_gen:
                        perf["decode_tok_s"] = float(m_gen.group(1))
                    if m_tokens:
                        perf["prompt_tokens"] = int(m_tokens.group(1))
                        perf["generated_tokens"] = int(m_tokens.group(2))
                    if m_drafts:
                        perf["drafts_accepted"] = m_drafts.group(1)
                    if m_gen or m_tokens:
                        perf["has_data"] = True
                        found_turn = True
                        break

        for line in reversed(lines):
            if "prompt" in line and "read in" in line:
                m_batch = re.search(r"(\d+) read in \d+ ms \(([\d\.]+) tok/s\)", line)
                if m_batch and int(m_batch.group(1)) >= 100:
                    perf["batch_prefill_tok_s"] = float(m_batch.group(2))
                    break

        for line in reversed(lines):
            if "decode expert cache hit rate:" in line:
                m_hit = re.search(r"hit rate: ([\d\.]+)%", line)
                if m_hit:
                    perf["expert_cache_hit"] = float(m_hit.group(1))
                    break

        mtime = os.path.getmtime(log_path)
        perf["is_active"] = (time.time() - mtime) < 10

        if found_turn:
            _last_known_inference_perf = perf.copy()
    except Exception:
        pass

    return perf


def discover_stored_models() -> list:
    """Discovers local .gguf model files across configured model storage directories."""
    seen = set()
    models = []
    for pattern in MODEL_STORAGE_DIRS:
        for directory in glob.glob(pattern.strip()):
            if not os.path.isdir(directory):
                continue
            try:
                for fname in sorted(os.listdir(directory)):
                    if not (fname.endswith(".gguf") or fname.endswith(".bin")):
                        continue
                    full_path = os.path.join(directory, fname)
                    if full_path in seen or not os.path.isfile(full_path):
                        continue
                    seen.add(full_path)
                    size_gb = round(os.path.getsize(full_path) / (1024**3), 2)
                    models.append({"name": fname, "size_gb": size_gb, "path": full_path})
            except Exception:
                continue
    return models


def get_system_metrics() -> dict:
    # 1. System RAM
    ram_info = {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                mem[parts[0].strip()] = int(parts[1].strip().split()[0])
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        ram_info = {
            "total_gb": round(total_kb / (1024**2), 1),
            "used_gb": round(used_kb / (1024**2), 1),
            "free_gb": round(avail_kb / (1024**2), 1),
            "percent": round((used_kb / total_kb) * 100, 1) if total_kb > 0 else 0.0,
        }
    except Exception:
        pass

    # 2. NVIDIA GPUs
    gpus = []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 10:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "temp_gpu": parts[2],
                        "power_draw": parts[3],
                        "power_limit": parts[4],
                        "clock_graphics": parts[5],
                        "clock_memory": parts[6],
                        "memory_used": parts[7],
                        "memory_total": parts[8],
                        "utilization_gpu": parts[9],
                        "bus_id": parts[10] if len(parts) > 10 else "",
                    }
                )
    except Exception:
        pass

    # 3. CPU Temperatures (hwmon)
    cpu_temps = {}
    for p in sorted(glob.glob("/sys/class/hwmon/hwmon*/temp*_input")):
        try:
            hw_dir = os.path.dirname(p)
            with open(os.path.join(hw_dir, "name"), "r", encoding="utf-8") as f:
                hw_name = f.read().strip()
            base = os.path.basename(p).replace("_input", "")
            label_p = os.path.join(hw_dir, f"{base}_label")
            if os.path.exists(label_p):
                with open(label_p, "r", encoding="utf-8") as f:
                    label = f.read().strip()
            else:
                label = base
            with open(p, "r", encoding="utf-8") as f:
                val = round(int(f.read().strip()) / 1000.0, 1)
            cpu_temps[f"{hw_name} - {label}"] = val
        except Exception:
            pass

    # 4. Inference Server Status (OpenAI-compatible /v1/models check)
    is_inference_serving = False
    is_inference_loading = False
    try:
        p_proc = subprocess.run(
            ["pgrep", "-f", "strata|llama-server|vllm"],
            capture_output=True,
            check=False,
        )
        is_proc_up = p_proc.returncode == 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.05)
            is_inference_serving = s.connect_ex((INFERENCE_HOST, INFERENCE_PORT)) == 0
        is_inference_loading = is_proc_up and not is_inference_serving
    except Exception:
        pass

    loaded_models = []
    if is_inference_serving:
        try:
            req_m = urllib.request.Request(
                f"http://{INFERENCE_HOST}:{INFERENCE_PORT}/v1/models"
            )
            with urllib.request.urlopen(req_m, timeout=0.8) as resp:
                m_json = json.loads(resp.read().decode("utf-8"))
                for m in m_json.get("data", []):
                    loaded_models.append(
                        {
                            "id": m.get("id", "Unknown"),
                            "status": m.get("status", {}).get("value", "loaded")
                            if isinstance(m.get("status"), dict)
                            else "loaded",
                            "context": m.get("meta", {}).get("n_ctx", 131072),
                            "modalities": m.get("architecture", {}).get(
                                "input_modalities", ["text"]
                            ),
                        }
                    )
        except Exception:
            pass

    watts = read_rapl_watts()

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_title": SERVER_TITLE,
        "server_subtitle": SERVER_SUBTITLE,
        "inference_port": INFERENCE_PORT,
        "base_system_watts": BASE_SYSTEM_WATTS,
        "enable_shutdown": ENABLE_SHUTDOWN_BUTTON,
        "power_cap_options": POWER_CAP_OPTIONS,
        "gpu1_offset_w": GPU1_POWER_OFFSET_W,
        "gpus": gpus,
        "ram": ram_info,
        "watts": watts,
        "temps": cpu_temps,
        "is_inference_serving": is_inference_serving,
        "is_inference_loading": is_inference_loading,
        "loaded_models": loaded_models,
        "stored_models": discover_stored_models(),
        "power_cap": get_active_power_cap(),
        "inference_perf": get_inference_perf(),
    }


# ==============================================================================
# Frontend Single-Page Dashboard
# ==============================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>v100AIServerMonitor — Live Telemetry Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0b0f19;
            --card-bg: rgba(18, 24, 38, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --status-green: #10b981;
            --status-green-bg: rgba(16, 185, 129, 0.15);
            --status-yellow: #f59e0b;
            --status-yellow-bg: rgba(245, 158, 11, 0.15);
            --status-red: #ef4444;
            --status-red-bg: rgba(239, 68, 68, 0.15);
            --accent-cyan: #06b6d4;
            --accent-orange: #f97316;
            --accent-purple: #a855f7;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at 15% 15%, #131b2e 0%, var(--bg-page) 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }

        .container { max-width: 1280px; margin: 0 auto; }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .title-area h1 { font-size: 1.7rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
        .title-area p { font-size: 0.9rem; color: var(--text-muted); margin-top: 4px; }

        .live-pill {
            display: flex;
            align-items: center;
            background: var(--status-green-bg);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--status-green);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .live-dot {
            width: 8px; height: 8px; background: var(--status-green);
            border-radius: 50%; margin-right: 8px;
            box-shadow: 0 0 10px var(--status-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .card-title { font-size: 1.1rem; font-weight: 700; color: #fff; }

        .state-tag {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 8px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .state-cool { background: var(--status-green-bg); color: var(--status-green); border: 1px solid rgba(16, 185, 129, 0.4); }
        .state-warm { background: var(--status-yellow-bg); color: var(--status-yellow); border: 1px solid rgba(245, 158, 11, 0.4); }
        .state-hot  { background: var(--status-red-bg); color: var(--status-red); border: 1px solid rgba(239, 68, 68, 0.4); }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 16px;
        }

        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 14px;
        }

        .stat-label {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .stat-val-huge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
        }

        .row-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.9rem;
        }
        .row-label { color: var(--text-muted); }
        .row-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; }

        .bar-wrap {
            width: 100%;
            height: 9px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            overflow: hidden;
            margin: 8px 0 14px 0;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
        }

        .model-card-item {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 14px;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            margin-bottom: 4px;
        }

        .peak-cap-card {
            display: flex;
            align-items: center;
            gap: 18px;
            flex-wrap: wrap;
            padding: 16px 22px;
            margin-bottom: 24px;
            border-left: 4px solid var(--accent-orange);
        }
        .peak-cap-sub { font-size: 0.78rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
        .peak-cap-btns { display: flex; gap: 10px; flex-wrap: wrap; }
        .peak-cap-btns button {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 20px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .peak-cap-btns button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.12); }
        .peak-cap-btns button:disabled { opacity: 0.45; cursor: wait; }
        .peak-cap-btns button.cap-active {
            background: rgba(249, 115, 22, 0.18);
            border-color: var(--accent-orange);
            color: var(--accent-orange);
            box-shadow: 0 0 12px rgba(249, 115, 22, 0.25);
        }
        .peak-cap-note { font-size: 0.75rem; color: var(--text-muted); max-width: 360px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-area">
                <h1 id="server-title">Dual Tesla V100 AI Server Monitor</h1>
                <p id="server-subtitle">Dual NVIDIA Tesla V100 PCIe (2x 32GB = 64GB HBM2) & Dual Intel Xeon Workstation</p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                <div class="live-pill" id="inference-status-pill" style="background: rgba(168, 85, 247, 0.12); border-color: rgba(168, 85, 247, 0.3); color: var(--accent-purple);">
                    <span>🧠 AI Engine: <strong id="inference-status-text">CHECKING...</strong></span>
                </div>
                <div class="live-pill" style="background: rgba(6, 182, 212, 0.12); border-color: rgba(6, 182, 212, 0.3); color: var(--accent-cyan);">
                    <span>⚡ V100 Total Power: <strong id="v100-total-power">-- W</strong></span>
                </div>
                <div class="live-pill">
                    <div class="live-dot"></div>
                    <span id="timestamp">Updating...</span>
                </div>
                <button id="btn-shutdown-server" style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #f87171; border-radius: 9999px; padding: 6px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; gap: 6px;">
                    <span>🛑 Shutdown Server</span>
                </button>
            </div>
        </header>

        <!-- PEAK POWER CAP SWITCHER -->
        <div class="card peak-cap-card">
            <div>
                <span class="card-title">⚡ Peak Power Cap (per V100)</span>
                <div class="peak-cap-sub" id="peak-cap-status">Loading current cap...</div>
            </div>
            <div class="peak-cap-btns" id="peak-cap-btns">
                <button data-w="140">140W</button>
                <button data-w="150">150W</button>
                <button data-w="160">160W</button>
                <button data-w="170">170W</button>
            </div>
            <span class="peak-cap-note">Sets the active baseline power cap for V100 compute cards. Works alongside v100-thermal-guard hysteresis protection.</span>
        </div>

        <div class="grid">
            <!-- GPU CARD 0: TESLA V100 #0 -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Tesla V100 #0 (32GB)</span>
                        <div id="v100-0-sub" style="font-size: 0.78rem; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Primary Compute • 32GB HBM2</div>
                    </div>
                    <span id="v100-0-badge" class="state-tag state-cool">COOL</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">GPU CORE TEMP</div>
                        <div class="stat-val-huge" id="v100-0-temp">-- °C</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">POWER DRAW</div>
                        <div class="stat-val-huge" id="v100-0-power">-- W</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Power Limit:</span><span class="row-val" id="v100-0-limit">-- W</span></div>
                <div class="row-item"><span class="row-label">GPU Utilization:</span><span class="row-val" id="v100-0-util">-- %</span></div>
                <div class="row-item"><span class="row-label">GPU Clock:</span><span class="row-val" id="v100-0-clock">-- MHz</span></div>
                <div class="row-item"><span class="row-label">HBM2 Memory Clock:</span><span class="row-val" id="v100-0-mclock">-- MHz</span></div>
                <div class="row-item"><span class="row-label">VRAM Used:</span><span class="row-val" id="v100-0-vram">-- / 32768 MiB</span></div>
                <div class="bar-wrap">
                    <div class="bar-fill" id="v100-0-vram-bar" style="width: 0%; background: var(--accent-cyan);"></div>
                </div>
            </div>

            <!-- GPU CARD 1: TESLA V100 #1 -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Tesla V100 #1 (32GB)</span>
                        <div id="v100-1-sub" style="font-size: 0.78rem; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Secondary Compute • 32GB HBM2</div>
                    </div>
                    <span id="v100-1-badge" class="state-tag state-cool">COOL</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">GPU CORE TEMP</div>
                        <div class="stat-val-huge" id="v100-1-temp">-- °C</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">POWER DRAW</div>
                        <div class="stat-val-huge" id="v100-1-power">-- W</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Power Limit:</span><span class="row-val" id="v100-1-limit">-- W</span></div>
                <div class="row-item"><span class="row-label">GPU Utilization:</span><span class="row-val" id="v100-1-util">-- %</span></div>
                <div class="row-item"><span class="row-label">GPU Clock:</span><span class="row-val" id="v100-1-clock">-- MHz</span></div>
                <div class="row-item"><span class="row-label">HBM2 Memory Clock:</span><span class="row-val" id="v100-1-mclock">-- MHz</span></div>
                <div class="row-item"><span class="row-label">VRAM Used:</span><span class="row-val" id="v100-1-vram">-- / 32768 MiB</span></div>
                <div class="bar-wrap">
                    <div class="bar-fill" id="v100-1-vram-bar" style="width: 0%; background: var(--accent-cyan);"></div>
                </div>
            </div>

            <!-- SYSTEM & RAM CARD -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">System & ECC Memory</span>
                        <div style="font-size: 0.78rem; color: var(--status-green); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Host System Memory & Display Adapter</div>
                    </div>
                    <span id="ram-badge" class="state-tag state-cool">ECC RAM</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">RAM USED</div>
                        <div class="stat-val-huge" id="ram-used">-- GB</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">RAM FREE</div>
                        <div class="stat-val-huge" id="ram-free">-- GB</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Total RAM:</span><span class="row-val" id="ram-total">-- GB</span></div>
                <div class="row-item"><span class="row-label">Memory Utilization:</span><span class="row-val" id="ram-pct">-- %</span></div>
                <div class="bar-wrap">
                    <div class="bar-fill" id="ram-bar" style="width: 0%; background: var(--accent-cyan);"></div>
                </div>
                <div class="row-item"><span class="row-label">Display GPU:</span><span class="row-val" id="display-gpu-temp">None / Headless</span></div>
            </div>

            <!-- CPU RAPL WATTS CARD -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Dual Xeon Power Draw</span>
                        <div style="font-size: 0.78rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Intel RAPL Hardware Energy Telemetry</div>
                    </div>
                    <span class="state-tag state-cool">RAPL LIVE</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">TOTAL CPU POWER</div>
                        <div class="stat-val-huge" id="total-cpu-watt">-- W</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">DRAM POWER</div>
                        <div class="stat-val-huge" id="ram-watt">-- W</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">CPU Socket 0 (package-0):</span><span class="row-val" id="cpu0-watt">-- W</span></div>
                <div class="row-item"><span class="row-label">CPU Socket 1 (package-1):</span><span class="row-val" id="cpu1-watt">-- W</span></div>
                <div class="row-item"><span class="row-label">CPU Package Temp:</span><span class="row-val" id="cpu-temp">-- °C</span></div>
            </div>

            <!-- TOTAL WORKSTATION POWER CONSUMPTION WIDGET -->
            <div class="card" style="border-left: 4px solid var(--accent-orange);">
                <div class="card-header">
                    <div>
                        <span class="card-title">Total Power Consumption</span>
                        <div style="font-size: 0.78rem; color: var(--accent-orange); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Combined Dual V100 + Dual Xeon + DRAM + System</div>
                    </div>
                    <span class="state-tag state-warm">MONITORED</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box" style="background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.25);">
                        <div class="stat-label" style="color: var(--accent-orange);">TOTAL LIVE POWER</div>
                        <div class="stat-val-huge" id="grand-total-power" style="color: var(--accent-orange);">-- W</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">DUAL V100 TOTAL</div>
                        <div class="stat-val-huge" id="v100-combined-power" style="color: var(--accent-cyan);">-- W</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Dual Tesla V100 GPUs:</span><span class="row-val" id="row-v100-power">-- W</span></div>
                <div class="row-item"><span class="row-label">Dual Xeon CPUs (RAPL):</span><span class="row-val" id="row-cpu-power">-- W</span></div>
                <div class="row-item"><span class="row-label">ECC DDR4 Memory (RAPL):</span><span class="row-val" id="row-dram-power">-- W</span></div>
                <div class="row-item"><span class="row-label">Estimated Motherboard & Fans:</span><span class="row-val" id="row-base-power">~45.0 W</span></div>
            </div>

            <!-- CARD 6: AI INFERENCE & TOKENS/SEC -->
            <div class="card" style="border-left: 4px solid var(--accent-purple);">
                <div class="card-header">
                    <div>
                        <span class="card-title">AI Inference Throughput</span>
                        <div id="inference-port-sub" style="font-size: 0.78rem; color: var(--accent-purple); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Tokens & Prefill Rate • Port 8086</div>
                    </div>
                    <span class="state-tag state-cool" id="inference-perf-badge">READY</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box" style="background: rgba(56, 189, 248, 0.08); border-color: rgba(56, 189, 248, 0.25);">
                        <div class="stat-label" style="color: var(--accent-cyan);">PREFILL SPEED</div>
                        <div class="stat-val-huge" id="perf-prefill" style="color: var(--accent-cyan);">-- tok/s</div>
                    </div>
                    <div class="stat-box" style="background: rgba(34, 197, 94, 0.08); border-color: rgba(34, 197, 94, 0.25);">
                        <div class="stat-label" style="color: var(--status-green);">DECODE SPEED</div>
                        <div class="stat-val-huge" id="perf-decode" style="color: var(--status-green);">-- tok/s</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Last Turn Tokens:</span><span class="row-val" id="perf-tokens">-- / --</span></div>
                <div class="row-item"><span class="row-label">MTP Speculation:</span><span class="row-val" id="perf-mtp" style="color: var(--accent-purple);">spec=3 (Optimal)</span></div>
                <div class="row-item"><span class="row-label">MoE Expert Cache Hit:</span><span class="row-val" id="perf-cache" style="color: var(--status-green);">100.0%</span></div>
            </div>
        </div>

        <!-- ACTIVE LOADED MODELS CARD -->
        <div class="card" style="margin-bottom: 24px;">
            <div class="card-header">
                <div>
                    <span class="card-title">Active Loaded AI Models in VRAM & RAM</span>
                    <div style="font-size: 0.78rem; color: var(--accent-purple); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">OpenAI-Compatible Inference Endpoint (/v1/models)</div>
                </div>
                <span id="models-badge-overall" class="state-tag state-cool">READY</span>
            </div>
            <div id="active-models-container">
                <div style="color: var(--text-muted); font-size: 0.9rem;">Checking active models...</div>
            </div>
        </div>

        <!-- DISCOVERED MODELS ON DISK -->
        <div class="card" style="margin-top: 10px; background: rgba(18, 24, 38, 0.45); border-color: rgba(255, 255, 255, 0.04);">
            <div class="card-header" style="margin-bottom: 8px;">
                <div>
                    <span class="card-title" style="font-size: 0.95rem; color: var(--text-muted);">💾 Discovered Model Weights on Local Storage</span>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">Auto-scanned from configured V100_MODEL_DIRS</div>
                </div>
                <span class="state-tag state-cool" style="font-size: 0.7rem;">LOCAL NVME</span>
            </div>
            <div id="stored-models-container" style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;"></div>
        </div>
    </div>

    <script>
        function getTempTag(temp) {
            if (temp < 60) return { text: 'COOL (Optimal) 🟢', cls: 'state-cool' };
            if (temp < 73) return { text: 'WARM (Active) 🟡', cls: 'state-warm' };
            return { text: 'HOT (Throttling) 🔴', cls: 'state-hot' };
        }

        function updateGpuCard(gpu, cardPrefix) {
            if (!gpu) return;
            const t = parseFloat(gpu.temp_gpu);
            const p = parseFloat(gpu.power_draw);
            const limit = parseFloat(gpu.power_limit);
            const memUsed = parseInt(gpu.memory_used, 10) || 0;
            const memTotal = parseInt(gpu.memory_total, 10) || 32768;
            const pct = Math.min(100, Math.round((memUsed / memTotal) * 100));

            if (gpu.bus_id) {
                const subEl = document.getElementById(cardPrefix + '-sub');
                if (subEl) subEl.textContent = `PCIe ${gpu.bus_id} • ${memTotal} MiB HBM2`;
            }

            const tempEl = document.getElementById(cardPrefix + '-temp');
            if (tempEl) tempEl.textContent = !isNaN(t) ? t + ' °C' : '-- °C';

            const powerEl = document.getElementById(cardPrefix + '-power');
            if (powerEl) powerEl.textContent = !isNaN(p) ? p.toFixed(1) + ' W' : '-- W';

            const limitEl = document.getElementById(cardPrefix + '-limit');
            if (limitEl) limitEl.textContent = !isNaN(limit) ? limit.toFixed(0) + ' W' : '-- W';

            const utilEl = document.getElementById(cardPrefix + '-util');
            if (utilEl) utilEl.textContent = (gpu.utilization_gpu ?? '0') + ' %';

            const clockEl = document.getElementById(cardPrefix + '-clock');
            if (clockEl) clockEl.textContent = (gpu.clock_graphics || '--') + ' MHz';

            const mclockEl = document.getElementById(cardPrefix + '-mclock');
            if (mclockEl) mclockEl.textContent = (gpu.clock_memory || '--') + ' MHz';

            const vramEl = document.getElementById(cardPrefix + '-vram');
            if (vramEl) vramEl.textContent = `${memUsed} / ${memTotal} MiB (${pct}%)`;

            const barEl = document.getElementById(cardPrefix + '-vram-bar');
            if (barEl) barEl.style.width = pct + '%';

            const badgeEl = document.getElementById(cardPrefix + '-badge');
            if (badgeEl && !isNaN(t)) {
                const tag = getTempTag(t);
                badgeEl.className = 'state-tag ' + tag.cls;
                badgeEl.textContent = tag.text;
            }
        }

        let activeCapWatts = null;
        let capBusy = false;

        function renderCapButtons() {
            document.querySelectorAll('#peak-cap-btns button').forEach(b => {
                const w = parseInt(b.dataset.w, 10);
                const active = activeCapWatts === w;
                b.classList.toggle('cap-active', active);
                if (!capBusy) b.textContent = w + 'W' + (active ? ' ✓' : '');
            });
        }

        async function setPowerCap(w) {
            if (capBusy) return;
            capBusy = true;
            const btns = Array.from(document.querySelectorAll('#peak-cap-btns button'));
            btns.forEach(b => { b.disabled = true; });
            const statusEl = document.getElementById('peak-cap-status');
            statusEl.textContent = 'Applying ' + w + 'W cap...';
            try {
                const res = await fetch('/api/power-cap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ watts: w })
                });
                const out = await res.json();
                if (res.ok && out.ok) {
                    activeCapWatts = out.watts;
                    statusEl.textContent = `Active baseline: ${out.watts}W (${(out.applied_gpus || []).join(', ') || 'saved'})`;
                } else {
                    statusEl.textContent = 'Failed to set cap: ' + (out.error || res.status);
                }
            } catch (e) {
                statusEl.textContent = 'Failed to set cap: ' + e;
            } finally {
                capBusy = false;
                btns.forEach(b => { b.disabled = false; });
                renderCapButtons();
            }
        }

        document.querySelectorAll('#peak-cap-btns button').forEach(b => {
            b.addEventListener('click', () => setPowerCap(parseInt(b.dataset.w, 10)));
        });

        const shutdownBtn = document.getElementById('btn-shutdown-server');
        if (shutdownBtn) {
            shutdownBtn.addEventListener('click', async () => {
                if (!confirm("Are you sure you want to shut down the server?")) return;
                shutdownBtn.disabled = true;
                shutdownBtn.innerHTML = "<span>⏳ Shutting down...</span>";
                try {
                    const res = await fetch('/api/shutdown', { method: 'POST' });
                    const data = await res.json();
                    if (data.ok) {
                        document.body.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100vh; background:#0f172a; color:#f87171; font-family:sans-serif; text-align:center;"><div><h1>🛑 Server is shutting down</h1><p style="color:#94a3b8; margin-top:8px;">System poweroff initiated safely. You can close this tab.</p></div></div>';
                    } else {
                        alert("Shutdown failed: " + (data.error || "Unknown error"));
                        shutdownBtn.disabled = false;
                        shutdownBtn.innerHTML = "<span>🛑 Shutdown Server</span>";
                    }
                } catch (e) {
                    document.body.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100vh; background:#0f172a; color:#f87171; font-family:sans-serif; text-align:center;"><div><h1>🛑 Server is shutting down</h1><p style="color:#94a3b8; margin-top:8px;">Connection closed.</p></div></div>';
                }
            });
        }

        let isFetching = false;
        async function fetchMetrics() {
            if (isFetching) return;
            isFetching = true;
            try {
                const res = await fetch('/api/metrics?t=' + Date.now(), { cache: 'no-store' });
                const data = await res.json();
                const perf = data.inference_perf || {};

                if (data.server_title) document.getElementById('server-title').textContent = data.server_title;
                if (data.server_subtitle) document.getElementById('server-subtitle').textContent = data.server_subtitle;
                if (data.enable_shutdown === false && shutdownBtn) shutdownBtn.style.display = 'none';

                document.getElementById('timestamp').textContent = data.timestamp;

                // Peak power cap state
                if (typeof data.power_cap === 'number' && !capBusy && data.power_cap !== activeCapWatts) {
                    activeCapWatts = data.power_cap;
                    renderCapButtons();
                    const capStatus = document.getElementById('peak-cap-status');
                    if (capStatus && capStatus.textContent.includes('Loading')) {
                        const offsetNote = data.gpu1_offset_w ? ` (GPU 1 offset: -${data.gpu1_offset_w}W)` : '';
                        capStatus.textContent = `Active baseline: ${data.power_cap}W${offsetNote}`;
                    }
                }

                // Tesla V100 GPUs
                const v100Gpus = (data.gpus || []).filter(g => g.name.toLowerCase().includes('v100'));
                const ComputeGpus = v100Gpus.length > 0 ? v100Gpus : (data.gpus || []);
                let totalV100Watts = 0.0;
                let v100WattsCount = 0;

                if (ComputeGpus.length > 0) {
                    updateGpuCard(ComputeGpus[0], 'v100-0');
                    const p0 = parseFloat(ComputeGpus[0].power_draw);
                    if (!isNaN(p0)) { totalV100Watts += p0; v100WattsCount++; }
                }
                if (ComputeGpus.length > 1) {
                    updateGpuCard(ComputeGpus[1], 'v100-1');
                    const p1 = parseFloat(ComputeGpus[1].power_draw);
                    if (!isNaN(p1)) { totalV100Watts += p1; v100WattsCount++; }
                }

                document.getElementById('v100-total-power').textContent =
                    v100WattsCount > 0 ? totalV100Watts.toFixed(1) + ' W' : '-- W';

                // RAM
                if (data.ram) {
                    document.getElementById('ram-used').textContent = data.ram.used_gb + ' GB';
                    document.getElementById('ram-free').textContent = data.ram.free_gb + ' GB';
                    document.getElementById('ram-total').textContent = data.ram.total_gb + ' GB';
                    document.getElementById('ram-pct').textContent = data.ram.percent + ' %';
                    document.getElementById('ram-bar').style.width = data.ram.percent + '%';
                    document.getElementById('ram-badge').textContent = Math.round(data.ram.total_gb) + ' GB RAM';
                }

                // RAPL Watts & CPU
                const w = data.watts || {};
                document.getElementById('total-cpu-watt').textContent = (w['Total CPU Watt'] ?? '--') + ' W';
                document.getElementById('cpu0-watt').textContent = (w['package-0'] ?? '--') + ' W';
                document.getElementById('cpu1-watt').textContent = (w['package-1'] ?? '--') + ' W';
                const dramWattVal = (w['dram'] || 0) + (w['dram-1'] || 0);
                document.getElementById('ram-watt').textContent = dramWattVal.toFixed(1) + ' W';

                const temps = data.temps || {};
                const cpuTemp = temps['coretemp - Package id 0'] || temps['coretemp - Package id 1'] || Object.values(temps)[0] || '--';
                document.getElementById('cpu-temp').textContent = cpuTemp + ' °C';

                // Total Power Consumption
                const cpuWattVal = parseFloat(w['Total CPU Watt']) || 0;
                const baseSystemW = data.base_system_watts || 45.0;
                const grandTotalW = totalV100Watts + cpuWattVal + dramWattVal + baseSystemW;

                document.getElementById('grand-total-power').textContent = grandTotalW.toFixed(1) + ' W';
                document.getElementById('v100-combined-power').textContent = totalV100Watts.toFixed(1) + ' W';
                document.getElementById('row-v100-power').textContent = totalV100Watts.toFixed(1) + ' W';
                document.getElementById('row-cpu-power').textContent = cpuWattVal.toFixed(1) + ' W';
                document.getElementById('row-dram-power').textContent = dramWattVal.toFixed(1) + ' W';
                document.getElementById('row-base-power').textContent = '~' + baseSystemW.toFixed(1) + ' W';

                // Display GPU (non-V100)
                const displayGpu = (data.gpus || []).find(g => !g.name.toLowerCase().includes('v100'));
                if (displayGpu) {
                    const temp = displayGpu.temp_gpu ? displayGpu.temp_gpu + ' °C' : '-- °C';
                    const mem = (displayGpu.memory_used && displayGpu.memory_total)
                        ? ` (${displayGpu.memory_used} / ${displayGpu.memory_total} MiB)` : '';
                    document.getElementById('display-gpu-temp').textContent = `${displayGpu.name}: ${temp}${mem}`;
                }

                // Inference Status Pill
                const infPill = document.getElementById('inference-status-pill');
                const infText = document.getElementById('inference-status-text');
                const port = data.inference_port || 8086;
                document.getElementById('inference-port-sub').textContent = `Tokens & Prefill Rate • Port ${port}`;

                if (data.is_inference_serving) {
                    infPill.style.background = 'rgba(34, 197, 94, 0.15)';
                    infPill.style.borderColor = 'rgba(34, 197, 94, 0.4)';
                    infPill.style.color = 'var(--status-green)';
                    infText.textContent = `ONLINE (Port ${port}) 🟢`;
                } else if (data.is_inference_loading) {
                    infPill.style.background = 'rgba(56, 189, 248, 0.15)';
                    infPill.style.borderColor = 'rgba(56, 189, 248, 0.4)';
                    infPill.style.color = 'var(--accent-cyan)';
                    infText.textContent = 'LOADING MODEL ⏳';
                } else {
                    infPill.style.background = 'rgba(148, 163, 184, 0.15)';
                    infPill.style.borderColor = 'rgba(148, 163, 184, 0.4)';
                    infPill.style.color = 'var(--text-muted)';
                    infText.textContent = 'STANDBY / OFFLINE';
                }

                // Active Loaded Models
                const modelsContainer = document.getElementById('active-models-container');
                const modelsBadge = document.getElementById('models-badge-overall');
                const loaded = data.loaded_models || [];

                if (data.is_inference_serving && loaded.length > 0) {
                    modelsBadge.className = 'state-tag state-cool';
                    modelsBadge.textContent = `ONLINE (${loaded.length} LOADED) ✅`;
                    let mHtml = '';
                    for (const m of loaded) {
                        const mods = (m.modalities || []).join(' + ').toUpperCase();
                        let prefillStr = 'Standby';
                        if (perf.prompt_cache_hit_pct >= 70) {
                            const rawRate = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + ' tok/s') : 'Cached';
                            prefillStr = `${rawRate} <span style="font-size:0.75rem; color:var(--status-green);">⚡ (${perf.prompt_cache_hit_pct.toFixed(1)}% hit)</span>`;
                        } else if (perf.prefill_tok_s > 0) {
                            prefillStr = perf.prefill_tok_s.toFixed(1) + ' tok/s';
                        } else if (perf.batch_prefill_tok_s > 0) {
                            prefillStr = perf.batch_prefill_tok_s.toFixed(1) + ' tok/s';
                        }
                        const decodeStr = perf.decode_tok_s > 0 ? perf.decode_tok_s.toFixed(1) + ' tok/s' : 'Standby';
                        const tokenStr = (perf.prompt_tokens || perf.generated_tokens)
                            ? `${perf.prompt_tokens.toLocaleString()} in / ${perf.generated_tokens.toLocaleString()} out`
                            : '0 in / 0 out';
                        const activePill = perf.is_active ? '<span class="state-tag state-warm" style="margin-left: 8px;">🔥 GENERATING</span>' : '';

                        mHtml += `<div class="model-card-item" style="border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.05);">
                            <div class="model-header">
                                <span style="color: var(--accent-cyan); font-size: 1.05rem; font-weight: 700;">🤖 ${m.id}${activePill}</span>
                                <span class="state-tag state-cool">${(m.status || 'loaded').toUpperCase()} 🟢</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px;">
                                <div class="row-item"><span class="row-label">⚡ Prefill Speed:</span><span class="row-val" style="color: var(--accent-cyan);">${prefillStr}</span></div>
                                <div class="row-item"><span class="row-label">🚀 Decode Speed:</span><span class="row-val" style="color: var(--status-green);">${decodeStr}</span></div>
                                <div class="row-item"><span class="row-label">📊 Last Turn Tokens:</span><span class="row-val">${tokenStr}</span></div>
                                <div class="row-item"><span class="row-label">🎯 MTP Draft Acceptance:</span><span class="row-val" style="color: var(--accent-purple);">${perf.drafts_accepted || 'spec=3'}</span></div>
                                <div class="row-item"><span class="row-label">🧠 MoE Cache Hit Rate:</span><span class="row-val" style="color: var(--status-green);">${(perf.expert_cache_hit || 100).toFixed(1)}%</span></div>
                                <div class="row-item"><span class="row-label">Max Context:</span><span class="row-val">${m.context ? m.context.toLocaleString() + ' tokens' : '131,072 tokens'}</span></div>
                                <div class="row-item"><span class="row-label">Modalities:</span><span class="row-val" style="color: var(--status-yellow);">${mods || 'TEXT'}</span></div>
                            </div>
                        </div>`;
                    }
                    modelsContainer.innerHTML = mHtml;
                } else if (data.is_inference_loading) {
                    modelsBadge.className = 'state-tag state-warm';
                    modelsBadge.textContent = 'LOADING MODEL ⏳';
                    modelsContainer.innerHTML = '<div style="padding: 12px; color: var(--accent-cyan); font-family: monospace;">⏳ Model weights and tensors are loading into VRAM & RAM...</div>';
                } else {
                    modelsBadge.className = 'state-tag state-hot';
                    modelsBadge.textContent = 'NO ACTIVE MODEL';
                    modelsContainer.innerHTML = `<div style="padding: 12px; color: var(--text-muted); font-family: monospace;">No active model detected on port ${port}.</div>`;
                }

                // Stored Models on Disk
                const storedContainer = document.getElementById('stored-models-container');
                const stored = data.stored_models || [];
                if (stored.length > 0) {
                    storedContainer.innerHTML = stored.map(m =>
                        `<div class="row-item" style="padding: 4px 0;">
                            <span class="row-label">📦 ${m.name}</span>
                            <span class="row-val" style="color: var(--status-green);">${m.size_gb} GB • Ready ✅</span>
                        </div>`
                    ).join('');
                } else {
                    storedContainer.innerHTML = '<div style="color: var(--text-muted);">No local .gguf files found in configured V100_MODEL_DIRS.</div>';
                }

                // Throughput summary card
                const prefillEl = document.getElementById("perf-prefill");
                const decodeEl = document.getElementById("perf-decode");
                const tokensEl = document.getElementById("perf-tokens");
                const mtpEl = document.getElementById("perf-mtp");
                const cacheEl = document.getElementById("perf-cache");
                const badgeEl = document.getElementById("inference-perf-badge");

                if (perf.prompt_cache_hit_pct >= 70) {
                    const rawRate = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + " tok/s") : "Cached";
                    prefillEl.innerHTML = `${rawRate} <span style="display:block; font-size:0.75rem; color:var(--status-green); font-weight:normal; margin-top:2px;">⚡ Cached (${perf.prompt_cache_hit_pct.toFixed(1)}% hit)</span>`;
                } else if (perf.prefill_tok_s > 0) {
                    prefillEl.textContent = perf.prefill_tok_s.toFixed(1) + " tok/s";
                } else {
                    prefillEl.textContent = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + " tok/s") : "Standby";
                }

                decodeEl.textContent = perf.decode_tok_s > 0 ? perf.decode_tok_s.toFixed(1) + " tok/s" : "Standby";
                tokensEl.textContent = (perf.prompt_tokens || perf.generated_tokens)
                    ? `${perf.prompt_tokens.toLocaleString()} in / ${perf.generated_tokens.toLocaleString()} out`
                    : "0 in / 0 out";
                mtpEl.textContent = perf.drafts_accepted || "spec=3 (Optimal)";
                cacheEl.textContent = (perf.expert_cache_hit || 100).toFixed(1) + "%";

                if (perf.is_active) {
                    badgeEl.className = "state-tag state-warm";
                    badgeEl.textContent = "🔥 ACTIVE INFERENCE";
                } else if (data.is_inference_serving) {
                    badgeEl.className = "state-tag state-cool";
                    badgeEl.textContent = "ONLINE / READY 🟢";
                } else {
                    badgeEl.className = "state-tag state-hot";
                    badgeEl.textContent = "OFFLINE";
                }
            } catch (err) {
                console.error(err);
            } finally {
                isFetching = false;
            }
        }

        setInterval(fetchMetrics, 1500);
        fetchMetrics();
    </script>
</body>
</html>
"""


# ==============================================================================
# HTTP Server Handler
# ==============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            data = get_system_metrics()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/api/shutdown"):
            if not ENABLE_SHUTDOWN_BUTTON:
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Shutdown disabled"}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": "Server shutting down now"}).encode("utf-8"))
            subprocess.Popen(["/sbin/shutdown", "-h", "now"])
            return

        if self.path.startswith("/api/power-cap"):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length).decode("utf-8") if length else ""
                watts = None
                try:
                    payload = json.loads(body) if body else {}
                    if isinstance(payload, dict) and "watts" in payload:
                        watts = int(float(payload["watts"]))
                except Exception:
                    watts = None

                if watts not in POWER_CAP_OPTIONS:
                    out = {
                        "ok": False,
                        "error": "watts must be one of " + "/".join(str(w) for w in POWER_CAP_OPTIONS),
                    }
                    code = 400
                else:
                    applied_w, applied_gpus = apply_power_cap(watts)
                    out = {"ok": True, "watts": applied_w, "applied_gpus": applied_gpus}
                    code = 200
            except Exception as e:
                out = {"ok": False, "error": str(e)}
                code = 500

            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    global PORT, BIND_HOST, INFERENCE_PORT, GPU1_POWER_OFFSET_W
    parser = argparse.ArgumentParser(
        description="v100AIServerMonitor — Live Web Telemetry Dashboard for Dual Tesla V100 Servers"
    )
    parser.add_argument("--host", default=BIND_HOST, help=f"Bind address (default: {BIND_HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Web dashboard port (default: {PORT})")
    parser.add_argument(
        "--inference-port",
        type=int,
        default=INFERENCE_PORT,
        help=f"Local OpenAI-compatible inference port to monitor (default: {INFERENCE_PORT})",
    )
    parser.add_argument(
        "--gpu1-offset",
        type=int,
        default=GPU1_POWER_OFFSET_W,
        help=f"Wattage offset subtracted from GPU 1 when setting power caps (default: {GPU1_POWER_OFFSET_W}W)",
    )
    args = parser.parse_args()

    BIND_HOST = args.host
    PORT = args.port
    INFERENCE_PORT = args.inference_port
    GPU1_POWER_OFFSET_W = args.gpu1_offset

    print(f"Starting v100AIServerMonitor on http://{BIND_HOST}:{PORT} ...")
    server = ThreadingHTTPServer((BIND_HOST, PORT), DashboardHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
