#!/usr/bin/env python3
import os
import re
import glob
import json
import time
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 9000

# Peak power cap switcher (shared baseline with v100-thermal-guard.py)
POWER_CAP_FILE = "/etc/v100-power-cap"
POWER_CAP_OPTIONS = [140, 150, 160, 170]
DEFAULT_POWER_CAP = 150


def read_bios_cooling():
    base = "/sys/class/firmware-attributes/hp-bioscfg/attributes"
    res = {"idle_fan_pct": "25", "pcie_fan_pct": "65"}
    try:
        p1 = os.path.join(base, "Increase Idle Fan Speed(%)", "current_value")
        if os.path.exists(p1):
            res["idle_fan_pct"] = open(p1).read().strip()
        p2 = os.path.join(base, "Increase PCIe Idle Fan Speed(%)", "current_value")
        if os.path.exists(p2):
            res["pcie_fan_pct"] = open(p2).read().strip()
    except Exception:
        pass
    return res

def get_active_power_cap():
    try:
        with open(POWER_CAP_FILE, "r") as f:
            return int(float(f.read().strip()))
    except Exception:
        return DEFAULT_POWER_CAP

def apply_power_cap(watts):
    """Set the peak power baseline: write the cap file for the thermal guard,
    then push it to V100 cards immediately (GPU 0 gets base watts, GPU 1 gets base - 10W)."""
    base_watts = int(watts)
    with open(POWER_CAP_FILE, "w") as f:
        f.write(str(base_watts) + "\n")
    applied = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and "V100" in parts[1].upper():
                idx_int = int(parts[0])
                # GPU 1 runs warmer, so give it 20W offset (minimum hardware limit is 100W)
                target_w = max(100, base_watts - 20) if idx_int == 1 else base_watts
                r = subprocess.run(
                    ["nvidia-smi", "-i", parts[0], "-pl", str(target_w)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if r.returncode == 0:
                    applied.append(f"{parts[0]}:{target_w}W")
    except Exception:
        pass
    return base_watts, applied

def read_rapl_watts():
    rapl_paths = glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
    names = {}
    for p in rapl_paths:
        name_path = os.path.join(os.path.dirname(p), "name")
        try:
            with open(name_path, "r") as f:
                names[p] = f.read().strip()
        except Exception:
            names[p] = os.path.basename(os.path.dirname(p))

    e1 = {}
    for p in rapl_paths:
        try:
            with open(p, "r") as f:
                e1[p] = int(f.read().strip())
        except Exception:
            pass

    time.sleep(0.1)

    e2 = {}
    for p in rapl_paths:
        try:
            with open(p, "r") as f:
                e2[p] = int(f.read().strip())
        except Exception:
            pass

    results = {}
    total_cpu = 0.0
    for p in e1:
        if p in e2:
            diff_uj = e2[p] - e1[p]
            watts = (diff_uj / 1e6) / 0.1
            nm = names.get(p, "unknown")
            results[nm] = round(watts, 2)
            if "package" in nm:
                total_cpu += watts

    results["Total CPU Watt"] = round(total_cpu, 2)
    return results

_last_known_strata_perf = {
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
    "has_data": False
}

def get_strata_perf():
    global _last_known_strata_perf
    candidate_logs = [
        "/home/user/strata/strata-ud-q4_k_xl.log",
        "/home/user/strata/strata-iq3_xxs.log",
        "/home/user/strata/strata.log"
    ]
    log_path = None
    latest_mtime = 0
    for cand in candidate_logs:
        if os.path.exists(cand):
            mt = os.path.getmtime(cand)
            if mt > latest_mtime:
                latest_mtime = mt
                log_path = cand
    if not log_path or not os.path.exists(log_path):
        return _last_known_strata_perf.copy()

    perf = _last_known_strata_perf.copy()
    try:
        # Read the last 300 lines so batch chatter does not push stats out of window
        with open(log_path, "r", errors="ignore") as f:
            lines = f.readlines()[-300:]
        
        found_turn = False
        for line in reversed(lines):
            if "strata serve: prompt" in line and "read in" in line:
                # Match full detailed turn with checkpoints
                m_details = re.search(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d\.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d\.]+) tok/s\)", line)
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
                        perf["prompt_cache_hit_pct"] = round((perf["reused_tokens"] / perf["prompt_tokens"]) * 100, 1)
                    perf["has_data"] = True
                    found_turn = True
                    break
                else:
                    m_prefill = re.search(r"read in \d+ ms \(([\d\.]+) tok/s\)", line)
                    m_gen = re.search(r"generated in \d+ ms \(([\d\.]+) tok/s\)", line)
                    m_tokens = re.search(r"prompt (\d+) tokens.*?, (\d+) generated", line)
                    m_drafts = re.search(r"drafts accepted (\d+ of \d+)", line)
                    if m_prefill: perf["prefill_tok_s"] = float(m_prefill.group(1))
                    if m_gen: perf["decode_tok_s"] = float(m_gen.group(1))
                    if m_tokens:
                        perf["prompt_tokens"] = int(m_tokens.group(1))
                        perf["generated_tokens"] = int(m_tokens.group(2))
                    if m_drafts: perf["drafts_accepted"] = m_drafts.group(1)
                    if m_gen or m_tokens:
                        perf["has_data"] = True
                        found_turn = True
                        break

        for line in reversed(lines):
            if "strata serve: prompt" in line and "read in" in line:
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
            _last_known_strata_perf = perf.copy()
    except Exception:
        pass

    return perf

def get_system_metrics():
    # RAM Usage
    ram_info = {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                mem[key] = int(val)
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        total_gb = round(total_kb / (1024**2), 1)
        used_gb = round(used_kb / (1024**2), 1)
        free_gb = round(avail_kb / (1024**2), 1)
        pct = round((used_kb / total_kb) * 100, 1) if total_kb > 0 else 0
        ram_info = {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "percent": pct
        }
    except Exception:
        pass

    gpus = []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "temp_gpu": parts[2],
                    "power_draw": parts[3],
                    "power_limit": parts[4],
                    "clock_graphics": parts[5],
                    "clock_memory": parts[6],
                    "memory_used": parts[7],
                    "memory_total": parts[8],
                    "utilization_gpu": parts[9] if len(parts) > 9 else "0"
                })
    except Exception:
        pass

    cpu_temps = {}
    for p in sorted(glob.glob("/sys/class/hwmon/hwmon*/temp*_input")):
        try:
            hw_dir = os.path.dirname(p)
            hw_name = open(os.path.join(hw_dir, "name")).read().strip()
            base = os.path.basename(p).replace("_input", "")
            label_p = os.path.join(hw_dir, f"{base}_label")
            label = open(label_p).read().strip() if os.path.exists(label_p) else base
            val = round(int(open(p).read().strip()) / 1000.0, 1)
            cpu_temps[f"{hw_name} - {label}"] = val
        except Exception:
            pass

    # Model 1: Strata models & MTP
    strata_status = {}
    for base_dir in ["/home/user/Strata-data/models", "/home/user/Strata-data/models/IQ3_S"]:
        if os.path.exists(base_dir):
            for f in os.listdir(base_dir):
                full_p = os.path.join(base_dir, f)
                if os.path.isfile(full_p):
                    sz_gb = round(os.path.getsize(full_p) / (1024**3), 2)
                    strata_status[f] = {"size_gb": sz_gb}

    # Track Strata MTP draft tensors (31 tensors, ~5.1 GB total)
    mtp_tensors_dir = "/home/user/Strata-data/mtp/tensors"
    mtp_tensor_count = 0
    mtp_total_mb = 0.0
    if os.path.exists(mtp_tensors_dir):
        files = os.listdir(mtp_tensors_dir)
        mtp_tensor_count = len(files)
        total_b = sum(os.path.getsize(os.path.join(mtp_tensors_dir, f)) for f in files)
        mtp_total_mb = round(total_b / (1024**2), 1)

    # Model 2: Qwen3.8-27B Dense models
    qwen27b_status = {}
    models27b_dir = "/home/user/models"
    if os.path.exists(models27b_dir):
        for f in os.listdir(models27b_dir):
            full_p = os.path.join(models27b_dir, f)
            if os.path.isfile(full_p):
                sz_gb = round(os.path.getsize(full_p) / (1024**3), 2)
                qwen27b_status[f] = {"size_gb": sz_gb}

    is_strata_downloading = False
    is_qwen27b_downloading = False
    is_mtp_downloading = False
    is_mtp_packing = False
    is_strata_serving = False
    try:
        p1 = subprocess.run(["pgrep", "-f", "curl.*Qwen3.8-Flash-Next"], capture_output=True)
        is_strata_downloading = (p1.returncode == 0)
        p2 = subprocess.run(["pgrep", "-f", "curl.*Qwen3.8-27B"], capture_output=True)
        is_qwen27b_downloading = (p2.returncode == 0)
        p3 = subprocess.run(["pgrep", "-f", "mtp_fetch.py"], capture_output=True)
        is_mtp_downloading = (p3.returncode == 0)
        p4 = subprocess.run(["pgrep", "-f", "mtp_pack.py"], capture_output=True)
        is_mtp_packing = (p4.returncode == 0)
        p5 = subprocess.run(["pgrep", "-f", "strata.*--serve|server.py.*--engine strata"], capture_output=True)
        is_proc_up = (p5.returncode == 0)
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)
        is_strata_serving = (s.connect_ex(('127.0.0.1', 8086)) == 0)
        s.close()
        is_strata_loading = is_proc_up and not is_strata_serving
    except Exception:
        pass

    loaded_models = []
    if is_strata_serving:
        try:
            import urllib.request
            req_m = urllib.request.Request('http://127.0.0.1:8086/v1/models')
            with urllib.request.urlopen(req_m, timeout=0.8) as resp:
                m_json = json.loads(resp.read().decode())
                for m in m_json.get('data', []):
                    loaded_models.append({
                        'id': m.get('id', 'Unknown'),
                        'status': m.get('status', {}).get('value', 'loaded'),
                        'context': m.get('meta', {}).get('n_ctx', 131072),
                        'modalities': m.get('architecture', {}).get('input_modalities', ['text'])
                    })
        except Exception:
            pass

    watts = read_rapl_watts()

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpus": gpus,
        "ram": ram_info,
        "watts": watts,
        "temps": cpu_temps,
        "strata_downloads": strata_status,
        "qwen27b_downloads": qwen27b_status,
        "is_strata_downloading": is_strata_downloading,
        "is_qwen27b_downloading": is_qwen27b_downloading,
        "is_mtp_downloading": is_mtp_downloading,
        "is_mtp_packing": is_mtp_packing,
        "is_strata_serving": is_strata_serving,
        "is_strata_loading": is_strata_loading,
        "loaded_models": loaded_models,
        "mtp_count": mtp_tensor_count,
        "mtp_mb": mtp_total_mb,
        "power_cap": get_active_power_cap(),
        "bios_cooling": read_bios_cooling(),
        "strata_perf": get_strata_perf()
    }

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HP Z8 G4 Live Monitor</title>
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

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
        }

        /* Thermal State Pill (Green / Yellow / Red) */
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
        .state-sval { background: var(--status-green-bg); color: var(--status-green); border: 1px solid rgba(16, 185, 129, 0.4); }
        .state-varm { background: var(--status-yellow-bg); color: var(--status-yellow); border: 1px solid rgba(245, 158, 11, 0.4); }
        .state-het  { background: var(--status-red-bg); color: var(--status-red); border: 1px solid rgba(239, 68, 68, 0.4); }

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

        /* Progress bars */
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

        .dl-card-item {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 14px;
        }
        .dl-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            margin-bottom: 4px;
        }
        .dl-name { font-weight: 700; color: #fff; }

        /* Peak power cap switcher strip */
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
        .peak-cap-note { font-size: 0.75rem; color: var(--text-muted); max-width: 320px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-area">
                <h1>HP Z8 G4 Server Monitor</h1>
                <p>Dual NVIDIA Tesla V100 PCIe (2x 32GB = 64GB VRAM) & Dual Intel Xeon Gold 6140</p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                <div class="live-pill" id="strata-status-pill" style="background: rgba(168, 85, 247, 0.12); border-color: rgba(168, 85, 247, 0.3); color: var(--accent-purple);">
                    <span>🧠 Strata AI: <strong id="strata-status-text">CHECKING...</strong></span>
                </div>
                <div class="live-pill" style="background: rgba(6, 182, 212, 0.12); border-color: rgba(6, 182, 212, 0.3); color: var(--accent-cyan);">
                    <span>⚡ V100 Total Power: <strong id="v100-total-power">-- W</strong></span>
                </div>
                <div class="live-pill" style="background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.3); color: var(--status-green);">
                    <span>💨 Cooling: <strong id="bios-cooling-pill">HP BIOS 25% Idle Fan (Custom Shroud Duct)</strong></span>
                </div>
                <div class="live-pill">
                    <div class="live-dot"></div>
                    <span id="timestamp">Updating...</span>
                </div>
                <button id="btn-shutdown-server" style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #f87171; border-radius: 9999px; padding: 6px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; gap: 6px;" onmouseover="this.style.background='rgba(239, 68, 68, 0.35)'; this.style.borderColor='#ef4444';" onmouseout="this.style.background='rgba(239, 68, 68, 0.15)'; this.style.borderColor='rgba(239, 68, 68, 0.4)';">
                    <span>🛑 Stäng Av Server</span>
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
            <span class="peak-cap-note">Sets the baseline cap for both V100s. Thermal guard keeps its safety throttling (73/75/78°C).</span>
        </div>

        <div class="grid">
            <!-- GPU CARD 0: TESLA V100 #0 -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Tesla V100 #0 (32GB)</span>
                        <div style="font-size: 0.78rem; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Primary Compute • PCIe 8D:00.0 • 32GB HBM2</div>
                    </div>
                    <span id="v100-0-badge" class="state-tag state-sval">COOL</span>
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
                <div class="row-item"><span class="row-label">Cooling:</span><span class="row-val bios-fan-val" style="color: var(--status-green);">Dedicated Shroud (BIOS 25% Min Fan)</span></div>
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
                        <div style="font-size: 0.78rem; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Secondary Compute • PCIe 99:00.0 • 32GB HBM2</div>
                    </div>
                    <span id="v100-1-badge" class="state-tag state-sval">COOL</span>
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
                <div class="row-item"><span class="row-label">Cooling:</span><span class="row-val bios-fan-val" style="color: var(--status-green);">Dedicated Shroud (BIOS 25% Min Fan)</span></div>
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
                        <div style="font-size: 0.78rem; color: var(--status-green); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">4x 16GB SK Hynix DDR4-2666</div>
                    </div>
                    <span id="ram-badge" class="state-tag state-sval">64 GB ECC</span>
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
                <div class="row-item"><span class="row-label">Total RAM:</span><span class="row-val" id="ram-total">64.0 GB</span></div>
                <div class="row-item"><span class="row-label">Memory Utilization:</span><span class="row-val" id="ram-pct">-- %</span></div>
                <div class="bar-wrap">
                    <div class="bar-fill" id="ram-bar" style="width: 0%; background: var(--accent-cyan);"></div>
                </div>
                                <div class="row-item"><span class="row-label">Display GPU (GT 1030):</span><span class="row-val" id="gt1030-temp">-- °C</span></div>
                <div class="row-item"><span class="row-label">NVMe SSD (NAND / ASIC):</span><span class="row-val" id="nvme-temp">-- °C</span></div>
            </div>

            <!-- CPU RAPL WATTS CARD -->
            <div class="card">
                <div class="card-header">
                    <div>
                        <span class="card-title">Dual Xeon Power Draw</span>
                        <div style="font-size: 0.78rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">2x Xeon Gold 6140 (36C / 72T)</div>
                    </div>
                    <span class="state-tag state-sval">RAPL LIVE</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">TOTAL CPU POWER</div>
                        <div class="stat-val-huge" id="total-cpu-watt">-- W</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">RAM POWER</div>
                        <div class="stat-val-huge" id="ram-watt">-- W</div>
                    </div>
                </div>
                <div class="row-item"><span class="row-label">Xeon 1 (Socket 0):</span><span class="row-val" id="cpu0-watt">-- W</span></div>
                <div class="row-item"><span class="row-label">Xeon 2 (Socket 1):</span><span class="row-val" id="cpu1-watt">-- W</span></div>
                <div class="row-item"><span class="row-label">CPU Package Temp:</span><span class="row-val" id="cpu-temp">-- °C</span></div>
            </div>

            <!-- TOTAL WORKSTATION POWER CONSUMPTION WIDGET -->
            <div class="card" style="border-left: 4px solid var(--accent-orange);">
                <div class="card-header">
                    <div>
                        <span class="card-title">Total Power Consumption</span>
                        <div style="font-size: 0.78rem; color: var(--accent-orange); font-family: `JetBrains Mono`, monospace; margin-top: 3px;">Combined Dual V100 + Dual Xeon + DRAM + System</div>
                    </div>
                    <span class="state-tag state-varm" id="power-efficiency-badge">MONITORED</span>
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

            <!-- CARD 6: STRATA AI INFERENCE & TOKENS/SEC (SLOT 6 IN GRID) -->
            <div class="card" style="border-left: 4px solid var(--accent-purple);">
                <div class="card-header">
                    <div>
                        <span class="card-title">Strata Inference Speed</span>
                        <div style="font-size: 0.78rem; color: var(--accent-purple); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Tokens & Prefill Rate • Port 8086</div>
                    </div>
                    <span class="state-tag state-sval" id="strata-perf-badge">READY</span>
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
                <div class="row-item"><span class="row-label">Last Turn Tokens:</span><span class="row-val" id="perf-tokens" style="font-family: monospace;">-- / --</span></div>
                <div class="row-item"><span class="row-label">MTP Speculation:</span><span class="row-val" id="perf-mtp" style="color: var(--accent-purple); font-family: monospace;">spec=3 (Optimal)</span></div>
                <div class="row-item"><span class="row-label">MoE Expert Cache Hit:</span><span class="row-val" id="perf-cache" style="color: var(--status-green); font-family: monospace;">100.0%</span></div>
                <div class="row-item"><span class="row-label">Tuned Config:</span><span class="row-val" style="color: var(--status-green);">spec=3 • 150W cap • 1380MHz</span></div>
            </div>
        </div>

        <!-- ACTIVE LOADED MODELS CARD -->
        <div class="card" style="margin-bottom: 24px;">
            <div class="card-header">
                <div>
                    <span class="card-title">Active Loaded AI Models in VRAM & RAM</span>
                    <div style="font-size: 0.78rem; color: var(--accent-purple); font-family: 'JetBrains Mono', monospace; margin-top: 3px;">Strata AI Engine • Port 8086 • Dual Tesla V100 64GB</div>
                </div>
                <span id="models-badge-overall" class="state-tag state-sval">READY</span>
            </div>
            <div id="active-models-container">
                <div style="color: var(--text-muted); font-size: 0.9rem;">Checking active models...</div>
            </div>
        </div>

        <!-- ACTIVE DOWNLOADS CARD (Shown ONLY when background downloads are active) -->
        <div class="card" id="active-downloads-card" style="margin-bottom: 24px; display: none;">
            <div class="card-header">
                <span class="card-title">⚡ Active Model Downloads in Progress</span>
                <span id="dl-badge-overall" class="state-tag state-varm">DOWNLOADING ⏳</span>
            </div>
            <div id="active-downloads-container"></div>
        </div>

        <!-- COMPLETED MODELS ARCHIVE -->
        <div class="card" style="margin-top: 10px; background: rgba(18, 24, 38, 0.45); border-color: rgba(255, 255, 255, 0.04);">
            <div class="card-header" style="margin-bottom: 8px;">
                <div>
                    <span class="card-title" style="font-size: 0.95rem; color: var(--text-muted);">💾 Installed Model Weights & Tensors on Disk</span>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">Stored locally in /home/user/Strata-data and /home/user/models</div>
                </div>
                <span class="state-tag state-sval" style="font-size: 0.7rem;">STORED ON NVME</span>
            </div>
            <div id="completed-models-container" style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;"></div>
        </div>


    </div>

    <script>
        function getTempTag(temp) {
            if (temp < 60) return { text: 'COOL (Optimal) 🟢', cls: 'state-sval' };
            if (temp < 75) return { text: 'WARM (Active) 🟡', cls: 'state-varm' };
            return { text: 'HOT (Needs Fan) 🔴', cls: 'state-het' };
        }

        function updateGpuCard(gpu, cardPrefix) {
            if (!gpu) return;
            const t = parseFloat(gpu.temp_gpu);
            const p = parseFloat(gpu.power_draw);
            const limit = parseFloat(gpu.power_limit);
            const memUsed = parseInt(gpu.memory_used, 10) || 0;
            const memTotal = parseInt(gpu.memory_total, 10) || 32768;
            const pct = Math.min(100, Math.round((memUsed / memTotal) * 100));

            const tempEl = document.getElementById(cardPrefix + '-temp');
            if (tempEl) tempEl.textContent = !isNaN(t) ? t + ' °C' : '-- °C';

            const powerEl = document.getElementById(cardPrefix + '-power');
            if (powerEl) powerEl.textContent = !isNaN(p) ? p.toFixed(1) + ' W' : (gpu.power_draw || '--') + ' W';

            const limitEl = document.getElementById(cardPrefix + '-limit');
            if (limitEl) limitEl.textContent = !isNaN(limit) ? limit.toFixed(0) + ' W' : (gpu.power_limit || '--') + ' W';

            const utilEl = document.getElementById(cardPrefix + '-util');
            if (utilEl) utilEl.textContent = (gpu.utilization_gpu !== undefined && gpu.utilization_gpu !== null ? gpu.utilization_gpu : '0') + ' %';

            const clockEl = document.getElementById(cardPrefix + '-clock');
            if (clockEl) clockEl.textContent = (gpu.clock_graphics || '--') + ' MHz';

            const mclockEl = document.getElementById(cardPrefix + '-mclock');
            if (mclockEl) mclockEl.textContent = (gpu.clock_memory || '--') + ' MHz';

            const vramEl = document.getElementById(cardPrefix + '-vram');
            if (vramEl) vramEl.textContent = memUsed + ' / ' + memTotal + ' MiB (' + pct + '%)';

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
                    statusEl.textContent = 'Active baseline: ' + out.watts + 'W per card (≈' + (out.watts * 2) + 'W peak dual-V100)';
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
                if (!confirm("⚠️ Är du säker på att du vill stänga av HP Z8 G4-servern helt?")) {
                    return;
                }
                shutdownBtn.disabled = true;
                shutdownBtn.innerHTML = "<span>⏳ Stänger av...</span>";
                shutdownBtn.style.background = "rgba(239, 68, 68, 0.5)";
                try {
                    const res = await fetch('/api/shutdown', { method: 'POST' });
                    const data = await res.json();
                    if (data.ok) {
                        alert("✅ Servern stängs nu av. Du kan stänga denna flik.");
                        document.body.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100vh; background:#0f172a; color:#f87171; font-family:sans-serif; text-align:center;"><div><h1>🛑 Servern har stängts av</h1><p style="color:#94a3b8;">Systemet stängs ner på ett säkert sätt. Du kan stänga webbläsarfliken.</p></div></div>';
                    } else {
                        alert("Kunde inte stänga av: " + (data.error || "Okänt fel"));
                        shutdownBtn.disabled = false;
                        shutdownBtn.innerHTML = "<span>🛑 Stäng Av Server</span>";
                    }
                } catch (e) {
                    // Om anslutningen bryts direkt pga avstängning
                    document.body.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100vh; background:#0f172a; color:#f87171; font-family:sans-serif; text-align:center;"><div><h1>🛑 Servern stängs av</h1><p style="color:#94a3b8;">Förbindelsen är bruten. Servern är nu avstängd.</p></div></div>';
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
                const perf = data.strata_perf || {};
                const tsEl = document.getElementById('timestamp');
                if (tsEl) tsEl.textContent = data.timestamp;

                // Peak power cap state
                if (typeof data.power_cap === 'number' && !capBusy && data.power_cap !== activeCapWatts) {
                    activeCapWatts = data.power_cap;
                    renderCapButtons();
                    const capStatus = document.getElementById('peak-cap-status');
                    if (capStatus && capStatus.textContent.indexOf('Loading') !== -1) {
                        capStatus.textContent = 'Active baseline: ' + data.power_cap + 'W per card (≈' + (data.power_cap * 2) + 'W peak dual-V100)';
                    }
                }

                // Tesla V100 GPUs (Show both GPU 0 and GPU 1)
                const v100Gpus = (data.gpus || []).filter(g => g.name.toLowerCase().includes('v100'));
                let totalV100Watts = 0.0;
                let v100WattsCount = 0;

                if (v100Gpus.length > 0) {
                    updateGpuCard(v100Gpus[0], 'v100-0');
                    updateGpuCard(v100Gpus[0], 'v100'); // Legacy fallback
                    const p0 = parseFloat(v100Gpus[0].power_draw);
                    if (!isNaN(p0)) { totalV100Watts += p0; v100WattsCount++; }
                }

                if (v100Gpus.length > 1) {
                    updateGpuCard(v100Gpus[1], 'v100-1');
                    const p1 = parseFloat(v100Gpus[1].power_draw);
                    if (!isNaN(p1)) { totalV100Watts += p1; v100WattsCount++; }
                }

                const totalWattsEl = document.getElementById('v100-total-power');
                if (totalWattsEl) {
                    totalWattsEl.textContent = v100WattsCount > 0 ? totalV100Watts.toFixed(1) + ' W' : '-- W';
                }

                // RAM
                if (data.ram) {
                    document.getElementById('ram-used').textContent = data.ram.used_gb + ' GB';
                    document.getElementById('ram-free').textContent = data.ram.free_gb + ' GB';
                    document.getElementById('ram-total').textContent = data.ram.total_gb + ' GB';
                    document.getElementById('ram-pct').textContent = data.ram.percent + ' %';
                    document.getElementById('ram-bar').style.width = data.ram.percent + '%';
                }

                // RAPL Watts & CPU
                const w = data.watts;
                document.getElementById('total-cpu-watt').textContent = (w['Total CPU Watt'] || '--') + ' W';
                document.getElementById('cpu0-watt').textContent = (w['package-0'] || '--') + ' W';
                document.getElementById('cpu1-watt').textContent = (w['package-1'] || '--') + ' W';
                document.getElementById('ram-watt').textContent = ((w['dram'] || 0) + (w['dram-1'] || 0)).toFixed(1) + ' W';

                const cpuTemp = data.temps['coretemp - Package id 0'] || data.temps['coretemp - Package id 1'] || '--';
                document.getElementById('cpu-temp').textContent = cpuTemp + ' °C';

                // Total Power Consumption Calculation
                let totalV100W = 0;
                if (data.gpus) {
                    data.gpus.forEach(g => {
                        if (g.name.includes("V100") && g.power_draw) {
                            const val = parseFloat(g.power_draw);
                            if (!isNaN(val)) totalV100W += val;
                        }
                    });
                }
                const cpuWattVal = parseFloat(w["Total CPU Watt"]) || 0;
                const dramWattVal = ((w["dram"] || 0) + (w["dram-1"] || 0));
                const baseSystemW = 45.0; // Base motherboard, chipset, fans, NVMe & GT 1030
                const grandTotalW = totalV100W + cpuWattVal + dramWattVal + baseSystemW;

                const grandTotalEl = document.getElementById("grand-total-power");
                if (grandTotalEl) grandTotalEl.textContent = grandTotalW.toFixed(1) + " W";
                const v100CombEl = document.getElementById("v100-combined-power");
                if (v100CombEl) v100CombEl.textContent = totalV100W.toFixed(1) + " W";
                const rowV100El = document.getElementById("row-v100-power");
                if (rowV100El) rowV100El.textContent = totalV100W.toFixed(1) + " W";
                const rowCpuEl = document.getElementById("row-cpu-power");
                if (rowCpuEl) rowCpuEl.textContent = cpuWattVal.toFixed(1) + " W";
                const rowDramEl = document.getElementById("row-dram-power");
                if (rowDramEl) rowDramEl.textContent = dramWattVal.toFixed(1) + " W";

                // GT 1030 Display GPU
                const gt = (data.gpus || []).find(g => g.name.includes('1030'));
                if (gt) {
                    const temp = gt.temp_gpu ? gt.temp_gpu + ' °C' : '-- °C';
                    const mem = (gt.memory_used && gt.memory_total) ? ' (' + gt.memory_used + ' / ' + gt.memory_total + ' MiB)' : '';
                    const gtEl = document.getElementById('gt1030-temp');
                    if (gtEl) gtEl.textContent = temp + mem;
                }

                                if (data.bios_cooling) {
                    const idlePct = data.bios_cooling.idle_fan_pct || '25';
                    const pillEl = document.getElementById('bios-cooling-pill');
                    if (pillEl) pillEl.textContent = 'HP BIOS ' + idlePct + '% Idle Fan (Custom Shroud Duct)';
                    document.querySelectorAll('.bios-fan-val').forEach(el => {
                        el.textContent = 'Dedicated Shroud (BIOS ' + idlePct + '% Min Fan)';
                    });
                }

                // Strata AI Status Pill
                const strataPill = document.getElementById('strata-status-pill');
                const strataText = document.getElementById('strata-status-text');
                if (data.is_strata_serving) {
                    strataPill.style.background = 'rgba(34, 197, 94, 0.15)';
                    strataPill.style.borderColor = 'rgba(34, 197, 94, 0.4)';
                    strataPill.style.color = 'var(--status-green)';
                    strataText.textContent = 'ONLINE / READY (Port 8086) 🟢';
                } else if (data.is_strata_loading) {
                    strataPill.style.background = 'rgba(56, 189, 248, 0.15)';
                    strataPill.style.borderColor = 'rgba(56, 189, 248, 0.4)';
                    strataPill.style.color = 'var(--accent-cyan)';
                    strataText.textContent = 'LOADING TO VRAM & RAM ⏳';
                } else if (data.is_mtp_packing) {
                    strataPill.style.background = 'rgba(234, 179, 8, 0.15)';
                    strataPill.style.borderColor = 'rgba(234, 179, 8, 0.4)';
                    strataPill.style.color = 'var(--status-yellow)';
                    strataText.textContent = 'PACKING MTP MODEL 📦';
                } else if (data.is_mtp_downloading) {
                    strataPill.style.background = 'rgba(168, 85, 247, 0.15)';
                    strataPill.style.borderColor = 'rgba(168, 85, 247, 0.4)';
                    strataPill.style.color = 'var(--accent-purple)';
                    strataText.textContent = 'DOWNLOADING MTP DRAFT ⏳';
                } else {
                    strataPill.style.background = 'rgba(148, 163, 184, 0.15)';
                    strataPill.style.borderColor = 'rgba(148, 163, 184, 0.4)';
                    strataPill.style.color = 'var(--text-muted)';
                    strataText.textContent = 'INITIALIZING...';
                }

                // Active Loaded Models
                const modelsContainer = document.getElementById('active-models-container');
                const modelsBadge = document.getElementById('models-badge-overall');
                const loaded = data.loaded_models || [];

                if (data.is_strata_serving && loaded.length > 0) {
                    modelsBadge.className = 'state-tag state-sval';
                    modelsBadge.textContent = 'ONLINE (' + loaded.length + ' LOADED) ✅';
                    let mHtml = '';
                    for (const m of loaded) {
                        const mods = (m.modalities || []).join(' + ').toUpperCase();
                        let prefillStr = 'Standby (~1,200 tok/s)';
                        if (perf.prompt_cache_hit_pct >= 70) {
                            const rawRate = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + ' tok/s') : '~1,180 tok/s';
                            prefillStr = `${rawRate} <span style="font-size:0.75rem; color:var(--status-green);">⚡ Cached (${perf.prompt_cache_hit_pct.toFixed(1)}% hit)</span>`;
                        } else if (perf.prefill_tok_s && perf.prefill_tok_s > 0) {
                            prefillStr = perf.prefill_tok_s.toFixed(1) + ' tok/s';
                        } else if (perf.batch_prefill_tok_s > 0) {
                            prefillStr = perf.batch_prefill_tok_s.toFixed(1) + ' tok/s';
                        }
                        const decodeStr = (perf.decode_tok_s && perf.decode_tok_s > 0) ? perf.decode_tok_s.toFixed(1) + ' tok/s' : 'Standby (~120 tok/s)';
                        const tokenStr = (perf.prompt_tokens || perf.generated_tokens) ? `${perf.prompt_tokens.toLocaleString()} in / ${perf.generated_tokens.toLocaleString()} out` : '0 in / 0 out';
                        const activePill = perf.is_active ? '<span class="state-tag state-varm" style="margin-left: 8px;">🔥 GENERATING</span>' : '';

                        mHtml += '<div class="dl-card-item" style="border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.05);">' +
                            '<div class="dl-header">' +
                                '<span class="dl-name" style="color: var(--accent-cyan); font-size: 1.1rem; font-weight: 700;">🤖 ' + m.id + activePill + '</span>' +
                                '<span class="state-tag state-sval">' + m.status.toUpperCase() + ' 🟢</span>' +
                            '</div>' +
                            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px;">' +
                                '<div class="row-item"><span class="row-label">⚡ Prefill Speed:</span><span class="row-val" style="color: var(--accent-cyan); font-weight: 700; font-family: monospace; font-size: 0.95rem;">' + prefillStr + '</span></div>' +
                                '<div class="row-item"><span class="row-label">🚀 Decode (Gen Speed):</span><span class="row-val" style="color: var(--status-green); font-weight: 700; font-family: monospace; font-size: 0.95rem;">' + decodeStr + '</span></div>' +
                                '<div class="row-item"><span class="row-label">📊 Last Turn Tokens:</span><span class="row-val" style="color: #fff; font-family: monospace;">' + tokenStr + '</span></div>' +
                                '<div class="row-item"><span class="row-label">🎯 MTP Draft Acceptance:</span><span class="row-val" style="color: var(--accent-purple);">' + (perf.drafts_accepted ? perf.drafts_accepted + ' (spec=3)' : 'spec=3 (Optimal)') + '</span></div>' +
                                '<div class="row-item"><span class="row-label">🧠 MoE Cache Hit Rate:</span><span class="row-val" style="color: var(--status-green);">' + (perf.expert_cache_hit || 100).toFixed(1) + '%</span></div>' +
                                '<div class="row-item"><span class="row-label">Max Context:</span><span class="row-val" style="color: #fff;">' + (m.context ? m.context.toLocaleString() + ' tokens' : '131,072 tokens') + '</span></div>' +
                                '<div class="row-item"><span class="row-label">Hardware Placement:</span><span class="row-val" style="color: var(--accent-purple);">Dual V100 (64GB VRAM) + NUMA RAM</span></div>' +
                                '<div class="row-item"><span class="row-label">Modalities:</span><span class="row-val" style="color: var(--status-yellow);">' + (mods || 'TEXT + IMAGE') + '</span></div>' +
                            '</div>' +
                        '</div>';
                    }
                    modelsContainer.innerHTML = mHtml;
                } else if (data.is_strata_loading) {
                    modelsBadge.className = 'state-tag state-varm';
                    modelsBadge.textContent = 'LOADING MODEL ⏳';
                    modelsContainer.innerHTML = `<div style="padding: 12px; color: var(--accent-cyan); font-family: \'JetBrains Mono\', monospace;">⏳ Model weights and tensors are currently being loaded into VRAM & RAM...</div>`;
                } else {
                    modelsBadge.className = 'state-tag state-het';
                    modelsBadge.textContent = 'NO ACTIVE MODEL';
                    modelsContainer.innerHTML = `<div style="padding: 12px; color: var(--text-muted); font-family: \'JetBrains Mono\', monospace;">No active model running on port 8086.</div>`;
                }

                // Downloads Logic: Only show Active Downloads Card if downloading
                const isDl = data.is_strata_downloading || data.is_qwen27b_downloading || data.is_mtp_downloading;
                const dlCard = document.getElementById('active-downloads-card');
                const dlContainer = document.getElementById('active-downloads-container');

                if (isDl) {
                    dlCard.style.display = 'block';
                    let activeHtml = '';
                    if (data.is_mtp_downloading) {
                        const mtpCount = data.mtp_count || 0;
                        const mtpMb = data.mtp_mb || 0;
                        const mtpPct = Math.min(100, Math.round((mtpCount / 31) * 100));
                        activeHtml += '<div class="dl-card-item">' +
                            '<div class="dl-header"><span class="dl-name">⚡ Strata MTP Draft Layer</span><span class="state-tag state-varm">FETCHING (' + mtpCount + '/31) ⏳</span></div>' +
                            '<div class="row-item"><span class="row-label">Progress:</span><span class="row-val">' + mtpCount + '/31 (' + mtpMb + ' MB)</span></div>' +
                            '<div class="bar-wrap"><div class="bar-fill" style="width: ' + mtpPct + '%; background: var(--accent-purple);"></div></div>' +
                        '</div>';
                    }
                    if (data.is_qwen27b_downloading) {
                        const q27 = data.qwen27b_downloads || {};
                        const q27_part = q27['Qwen3.8-27B-UD-Q6_K_XL.gguf.part'];
                        const q27_sz = q27_part ? q27_part.size_gb : 0;
                        const q27_pct = Math.min(100, Math.round((q27_sz / 23.56) * 100));
                        activeHtml += '<div class="dl-card-item">' +
                            '<div class="dl-header"><span class="dl-name">Qwen3.8-27B-UD-Q6_K_XL.gguf</span><span class="state-tag state-varm">DOWNLOADING ⏳</span></div>' +
                            '<div class="row-item"><span class="row-label">Size:</span><span class="row-val">' + q27_sz + ' GB / 23.56 GB (' + q27_pct + '%)</span></div>' +
                            '<div class="bar-wrap"><div class="bar-fill" style="width: ' + q27_pct + '%; background: var(--status-yellow);"></div></div>' +
                        '</div>';
                    }
                    dlContainer.innerHTML = activeHtml;
                } else {
                    dlCard.style.display = 'none';
                }

                // Completed Models Archive (Clean bottom list)
                const compContainer = document.getElementById('completed-models-container');
                if (compContainer) {
                    compContainer.innerHTML = 
                        '<div class="row-item" style="padding: 4px 0;"><span class="row-label">⚡ Strata Qwen3.8-Flash-Next IQ3_S (Shard 1/2)</span><span class="row-val" style="color: var(--status-green);">54.8 GB • Ready ✅</span></div>' +
                        '<div class="row-item" style="padding: 4px 0;"><span class="row-label">⚡ Strata Qwen3.8-Flash-Next IQ3_S (Shard 2/2)</span><span class="row-val" style="color: var(--status-green);">28.8 GB • Ready ✅</span></div>' +
                        '<div class="row-item" style="padding: 4px 0;"><span class="row-label">⚡ Strata MTP Speculative Draft Tensors (31/31)</span><span class="row-val" style="color: var(--status-green);">4,972.7 MB • Ready ✅</span></div>' +
                        '<div class="row-item" style="padding: 4px 0;"><span class="row-label">📦 Qwen3.8-27B-UD-Q6_K_XL.gguf (Standalone Dense)</span><span class="row-val" style="color: var(--accent-cyan);">23.56 GB • Ready ✅</span></div>';
                }

                // Update bottom strata-perf-card
                const prefillEl = document.getElementById("perf-prefill");
                const decodeEl = document.getElementById("perf-decode");
                const tokensEl = document.getElementById("perf-tokens");
                const mtpEl = document.getElementById("perf-mtp");
                const cacheEl = document.getElementById("perf-cache");
                const badgeEl = document.getElementById("strata-perf-badge");

                if (prefillEl && decodeEl && tokensEl && mtpEl && cacheEl) {
                    if (perf.prompt_cache_hit_pct >= 70) {
                        const rawRate = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + " tok/s") : "~1,180 tok/s";
                        prefillEl.innerHTML = rawRate + ` <span style="display:block; font-size:0.75rem; color:var(--status-green); font-weight:normal; margin-top:2px;">⚡ Cached (${perf.prompt_cache_hit_pct.toFixed(1)}% hit)</span>`;
                    } else if (perf.prefill_tok_s && perf.prefill_tok_s > 0) {
                        prefillEl.textContent = perf.prefill_tok_s.toFixed(1) + " tok/s";
                    } else {
                        prefillEl.textContent = perf.batch_prefill_tok_s > 0 ? (perf.batch_prefill_tok_s.toFixed(1) + " tok/s") : "Standby (~1,200 tok/s)";
                    }

                    decodeEl.textContent = (perf.decode_tok_s && perf.decode_tok_s > 0) ? perf.decode_tok_s.toFixed(1) + " tok/s" : "Standby (~120 tok/s)";
                    tokensEl.textContent = (perf.prompt_tokens || perf.generated_tokens) ? (perf.prompt_tokens.toLocaleString() + " in / " + perf.generated_tokens.toLocaleString() + " out") : "0 in / 0 out";
                    mtpEl.textContent = (perf.drafts_accepted && perf.drafts_accepted !== "N/A") ? (perf.drafts_accepted + " (spec=3)") : "spec=3 (Optimal)";
                    cacheEl.textContent = (perf.expert_cache_hit || 100).toFixed(1) + "%";

                    if (badgeEl) {
                        if (perf.is_active) {
                            badgeEl.className = "state-tag state-varm";
                            badgeEl.textContent = "🔥 ACTIVE INFERENCE";
                        } else if (data.is_strata_serving) {
                            badgeEl.className = "state-tag state-sval";
                            badgeEl.textContent = "ONLINE / READY 🟢";
                        } else {
                            badgeEl.className = "state-tag state-het";
                            badgeEl.textContent = "OFFLINE";
                        }
                    }
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

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            data = get_system_metrics()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/api/shutdown"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": "Server shutting down now"}).encode("utf-8"))
            # Trigger controlled shutdown in background after response is sent
            subprocess.Popen(["/bin/bash", "-c", "sleep 1 && echo 'Mika21sj.' | sudo -S shutdown -h now"])
            return
        elif self.path.startswith("/api/power-cap"):
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
                if watts is None and "?" in self.path:
                    for kv in self.path.split("?", 1)[1].split("&"):
                        if kv.startswith("watts="):
                            try:
                                watts = int(float(kv.split("=", 1)[1]))
                            except Exception:
                                watts = None
                if watts not in POWER_CAP_OPTIONS:
                    out = {"ok": False, "error": "watts must be one of " + "/".join(str(w) for w in POWER_CAP_OPTIONS)}
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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    server.serve_forever()
