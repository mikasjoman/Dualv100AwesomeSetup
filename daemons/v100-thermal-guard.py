#!/usr/bin/env python3
"""
HP Z8 G4 Workstation - Tesla V100 Dual GPU Thermal & Dynamic Eco Daemon
- Holds active power cap at the baseline in /etc/v100-power-cap (default 150W,
  switchable live from the dashboard's 140/150/160/170W peak-cap buttons).
- Idles down to hardware minimum (100W, 135MHz) when inactive.
- Multi-stage thermal hysteresis with voltage/clock step-down:
  * Stage 0 (Normal < 73°C): Baseline power cap, 1380 MHz graphics clock
  * Stage 1 (>= 73°C, recovers <= 68°C): 115W cap, 1230 MHz graphics clock
  * Stage 2 (>= 75°C, recovers <= 71°C to Stage 1): 100W cap, 1140 MHz graphics clock
- ABORT PROTECTION: If any GPU reaches >= 80°C, automatically terminates (SIGTERM/SIGKILL)
  the compute processes running on that GPU to protect hardware.
- Strictly manages V100 cards and isolates the GT 1030.
Logs to /var/log/v100-thermal-guard.log
"""

import subprocess
import time
import sys
import os
import signal
import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = "/var/log/v100-thermal-guard.log"

try:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
except Exception:
    handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger("V100PowerGuard")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Thermal and Power Policy Constants
POWER_CAP_FILE = "/etc/v100-power-cap"   # live baseline active cap, switchable from the dashboard
DEFAULT_ACTIVE_POWER_LIMIT_W = 150
ACTIVE_POWER_LIMIT_W = DEFAULT_ACTIVE_POWER_LIMIT_W  # refreshed from POWER_CAP_FILE each loop
IDLE_TIMEOUT_SEC      = 120   # Time in seconds with 0% utilization before entering idle eco mode
ABORT_TEMP            = 80.0  # °C -> abort/kill any processes running on the GPU!
THROTTLE_TEMP_STAGE1  = 73.0  # °C -> throttle to 115W and 1230 MHz
THROTTLE_POWER_STAGE1 = 115
THROTTLE_CLOCK_STAGE1 = 1230
THROTTLE_TEMP_STAGE2  = 75.0  # °C -> throttle to 100W and 1140 MHz
THROTTLE_POWER_STAGE2 = 100
THROTTLE_CLOCK_STAGE2 = 1140
RECOVERY_TEMP_STAGE2  = 71.0  # °C -> step back up from Stage 2 (100W) to Stage 1 (115W)
RECOVERY_TEMP_STAGE1  = 68.0  # °C -> step back up from Stage 1 (115W) to Stage 0 (Active Baseline)
ACTIVE_CLOCK_MHZ      = 1380
ECO_CLOCK_MHZ         = 135


def read_power_cap_file():
    """Reads the live active power-cap baseline (set by the dashboard buttons)."""
    try:
        with open(POWER_CAP_FILE, "r") as f:
            w = float(f.read().strip())
        if 100.0 <= w <= 300.0:
            return w
        logger.warning(f"Power cap file value {w}W out of range 100-300W; ignoring.")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not read {POWER_CAP_FILE}: {e}")
    return None


def run_command(cmd):
    """Executes a command safely without throwing unhandled exceptions."""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_gpu_min_power_limit(gpu_index):
    """Queries hardware VBIOS minimum allowable power limit to prevent out-of-range errors."""
    out = run_command(["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "POWER"])
    if out:
        for line in out.splitlines():
            if "Min Power Limit" in line and ":" in line:
                val = line.split(":", 1)[1].replace("W", "").strip()
                try:
                    return float(val)
                except ValueError:
                    pass
    return 100.0  # Conservative fallback min limit for Tesla V100 PCIe


def get_v100_gpus():
    """
    Returns list of dicts for V100 cards only (0, 1, or 2 cards).
    Strictly ignores the GT 1030 or any other display adapter.
    """
    out = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,temperature.gpu,power.draw,power.limit,utilization.gpu",
        "--format=csv,noheader,nounits"
    ])
    if not out:
        return []

    v100s = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 7:
            idx, name, uuid, temp_str, draw_str, limit_str, util_str = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            )
            if "V100" in name.upper():
                try:
                    v100s.append({
                        'index': idx,
                        'name': name,
                        'uuid': uuid,
                        'temp': float(temp_str),
                        'draw': float(draw_str),
                        'limit': float(limit_str),
                        'util': float(util_str)
                    })
                except ValueError:
                    continue
    return v100s


def set_power_cap(gpu_index, watts):
    logger.info(f"Applying power limit {watts:.0f}W to Tesla V100 (GPU Index {gpu_index})")
    run_command(["nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))])


def set_gpu_clock(gpu_index, mhz):
    logger.info(f"Applying graphics clock lock {mhz} MHz to Tesla V100 (GPU Index {gpu_index})")
    run_command(["nvidia-smi", "-i", str(gpu_index), "-lgc", f"{int(mhz)},{int(mhz)}"])


def abort_gpu_processes(gpu_index, temp):
    """
    Kills any compute applications running on the overheated GPU to bring temperature down immediately.
    """
    logger.critical(f"GPU {gpu_index} REACHED {temp:.1f}°C (>= {ABORT_TEMP}°C)! INITIATING EMERGENCY ABORT!")
    out = run_command([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits"
    ])
    if not out:
        logger.warning(f"No compute processes returned by query-compute-apps on GPU {gpu_index}.")
        return

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 1 and parts[0].isdigit():
            pid = int(parts[0])
            pname = parts[1] if len(parts) > 1 else "unknown"
            logger.critical(f"ABORTING process PID {pid} ({pname}) on GPU {gpu_index} due to thermal cutoff!")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                # If still alive, SIGKILL
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.error(f"Error terminating PID {pid}: {e}")


def main():
    global ACTIVE_POWER_LIMIT_W
    logger.info("Starting Tesla V100 Dynamic Eco & Thermal Guard Daemon (Stage1: 73°C/115W/1230MHz, Stage2: 75°C/100W/1140MHz, Abort: >= 80°C)...")
    run_command(["nvidia-smi", "-pm", "1"])

    gpu_state = {}           # index -> {'current_cap', 'current_clock', 'min_limit', 'idle_since', 'is_eco', 'stage'}
    last_log_time = 0
    LOG_INTERVAL_SEC = 120   # Status heartbeat every 2 minutes
    no_card_logged = False

    while True:
        try:
            v100_list = get_v100_gpus()

            if not v100_list:
                if not no_card_logged:
                    logger.info("No Tesla V100 cards detected currently. Sleeping and polling...")
                    no_card_logged = True
                time.sleep(10)
                continue

            if no_card_logged:
                logger.info(f"Tesla V100 card(s) detected! Active count: {len(v100_list)}")
                no_card_logged = False
                run_command(["nvidia-smi", "-pm", "1"])

            current_time = time.time()
            log_heartbeat = (current_time - last_log_time) >= LOG_INTERVAL_SEC

            # Re-read the switchable active baseline (dashboard peak-cap buttons)
            cap_now = read_power_cap_file()
            if cap_now is not None and abs(cap_now - ACTIVE_POWER_LIMIT_W) > 0.01:
                logger.info(f"Active power baseline switched: {ACTIVE_POWER_LIMIT_W:.0f}W -> {cap_now:.0f}W ({POWER_CAP_FILE})")
                ACTIVE_POWER_LIMIT_W = cap_now

            for gpu in v100_list:
                idx = gpu['index']
                temp = gpu['temp']
                current_limit = gpu['limit']
                draw = gpu['draw']
                util = gpu['util']

                # Initialize GPU state
                if idx not in gpu_state:
                    hw_min = get_gpu_min_power_limit(idx)
                    gpu_baseline_w = ACTIVE_POWER_LIMIT_W
                    init_stage = 2 if temp >= THROTTLE_TEMP_STAGE2 else (1 if temp >= THROTTLE_TEMP_STAGE1 else 0)
                    init_clock = THROTTLE_CLOCK_STAGE2 if init_stage == 2 else (THROTTLE_CLOCK_STAGE1 if init_stage == 1 else ACTIVE_CLOCK_MHZ)
                    init_cap = max(THROTTLE_POWER_STAGE2, hw_min) if init_stage == 2 else (max(THROTTLE_POWER_STAGE1, hw_min) if init_stage == 1 else gpu_baseline_w)
                    gpu_state[idx] = {
                        'current_cap': init_cap,
                        'current_clock': init_clock,
                        'min_limit': hw_min,
                        'idle_since': current_time,
                        'is_eco': False,
                        'stage': init_stage
                    }
                    logger.info(f"Initialized GPU {idx} ({gpu['name']}): Temp={temp:.0f}°C, Stage={init_stage}, Cap={init_cap:.0f}W, Clock={init_clock}MHz")
                    set_power_cap(idx, init_cap)
                    set_gpu_clock(idx, init_clock)

                state = gpu_state[idx]
                hw_min = state['min_limit']
                gpu_baseline_w = ACTIVE_POWER_LIMIT_W

                # 1. Idle Detection & Dynamic Eco Capping
                if util < 5.0 and temp < THROTTLE_TEMP_STAGE1:
                    if (current_time - state['idle_since']) >= IDLE_TIMEOUT_SEC:
                        if not state['is_eco']:
                            logger.info(f"GPU {idx} idle for {IDLE_TIMEOUT_SEC}s -> Entering Eco Idle Mode ({hw_min:.0f}W cap, {ECO_CLOCK_MHZ}MHz clock)")
                            state['is_eco'] = True
                            state['stage'] = 0
                else:
                    if util >= 5.0:
                        state['idle_since'] = current_time
                    if state['is_eco']:
                        logger.info(f"GPU {idx} active compute detected (util={util:.0f}%) -> Exiting Eco Idle Mode")
                        state['is_eco'] = False

                # 2. Thermal State Machine with Hysteresis
                prev_stage = state['stage']
                if temp >= ABORT_TEMP:
                    abort_gpu_processes(idx, temp)
                    state['stage'] = 2
                elif temp >= THROTTLE_TEMP_STAGE2:
                    if prev_stage < 2:
                        logger.warning(f"GPU {idx} HIGH TEMP: {temp:.0f}°C (>= {THROTTLE_TEMP_STAGE2:.0f}°C)! Entering Stage 2 thermal throttle ({THROTTLE_POWER_STAGE2}W, {THROTTLE_CLOCK_STAGE2}MHz).")
                    state['stage'] = 2
                elif temp >= THROTTLE_TEMP_STAGE1:
                    if prev_stage == 0:
                        logger.warning(f"GPU {idx} ELEVATED TEMP: {temp:.0f}°C (>= {THROTTLE_TEMP_STAGE1:.0f}°C). Entering Stage 1 thermal throttle ({THROTTLE_POWER_STAGE1}W, {THROTTLE_CLOCK_STAGE1}MHz).")
                        state['stage'] = 1
                    # If prev_stage == 2, stay in Stage 2 until temp <= RECOVERY_TEMP_STAGE2 (71°C)!
                elif temp <= RECOVERY_TEMP_STAGE1:
                    if prev_stage > 0:
                        logger.info(f"GPU {idx} cooled down to {temp:.0f}°C (<= {RECOVERY_TEMP_STAGE1:.0f}°C). Restoring Stage 0 Active Baseline ({gpu_baseline_w:.0f}W, {ACTIVE_CLOCK_MHZ}MHz).")
                        state['stage'] = 0
                elif temp <= RECOVERY_TEMP_STAGE2:
                    if prev_stage == 2:
                        logger.info(f"GPU {idx} cooled down to {temp:.0f}°C (<= {RECOVERY_TEMP_STAGE2:.0f}°C). Stepping from Stage 2 up to Stage 1 ({THROTTLE_POWER_STAGE1}W, {THROTTLE_CLOCK_STAGE1}MHz).")
                        state['stage'] = 1

                # 3. Compute Target Power Cap and Graphics Clock
                if state['is_eco']:
                    target_cap = hw_min
                    target_clock = ECO_CLOCK_MHZ
                elif state['stage'] == 2:
                    target_cap = max(THROTTLE_POWER_STAGE2, hw_min)
                    target_clock = THROTTLE_CLOCK_STAGE2
                elif state['stage'] == 1:
                    target_cap = max(THROTTLE_POWER_STAGE1, hw_min)
                    target_clock = THROTTLE_CLOCK_STAGE1
                else:
                    target_cap = gpu_baseline_w
                    target_clock = ACTIVE_CLOCK_MHZ

                # Apply power cap change only when needed
                if abs(current_limit - target_cap) > 2 or abs(state['current_cap'] - target_cap) > 2:
                    set_power_cap(idx, target_cap)
                    state['current_cap'] = target_cap

                # Apply clock change only when needed
                if state['current_clock'] != target_clock:
                    set_gpu_clock(idx, target_clock)
                    state['current_clock'] = target_clock

                if log_heartbeat:
                    mode = "ECO" if state['is_eco'] else f"ACTIVE-S{state['stage']}"
                    logger.info(f"GPU {idx} [{mode}]: Temp={temp:.0f}°C, Draw={draw:.1f}W, Util={util:.0f}%, Cap={state['current_cap']:.0f}W, Clock={state['current_clock']}MHz")

            if log_heartbeat:
                last_log_time = current_time

            time.sleep(2)  # Check every 2 seconds for fast safety response

        except Exception as err:
            logger.error(f"Watchdog loop exception: {err}")
            time.sleep(5)


if __name__ == "__main__":
    main()
