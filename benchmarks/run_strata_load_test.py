import urllib.request
import json
import time
import sys
import subprocess

SERVER_IP = "192.168.1.240"
PORT = 8086
SAFETY_TEMP_LIMIT = 77.0
NUM_ITERATIONS = 5

prompt = (
    "Detailed analysis of high-performance computing on modern accelerators. "
    "Explain in exhaustive depth the mathematical foundations of deep learning architectures, "
    "specifically Mixture-of-Experts (MoE) routing mechanisms, sparse gating topologies, "
    "K-V cache memory bandwidth bottlenecks, speculative decoding techniques with draft models, "
    "NUMA memory domains across dual CPU sockets, and thermal throttling in passive workstation server chassis."
)

def get_gpu_telemetry():
    res = subprocess.check_output(
        ["ssh", f"user@{SERVER_IP}", "nvidia-smi -i 0,1 --query-gpu=index,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits"],
        text=True
    ).strip().splitlines()
    data = {}
    for line in res:
        parts = [p.strip() for p in line.split(",")]
        data[int(parts[0])] = {
            "temp": float(parts[1]),
            "pwr": float(parts[2]),
            "util": float(parts[3])
        }
    return data

print("=" * 70)
print(f"STARTAR STRATA GPU LOAD TEST PÅ DUAL TESLA V100")
print(f"Säkerhetsgräns: {SAFETY_TEMP_LIMIT}°C (Avbryter direkt om någon GPU når {SAFETY_TEMP_LIMIT}°C)")
print("=" * 70)
sys.stdout.flush()

start_test_time = time.time()
stopped_early = False

for i in range(1, NUM_ITERATIONS + 1):
    print(f"\n--- Körning {i}/{NUM_ITERATIONS} startar ---")
    sys.stdout.flush()
    
    url = f"http://{SERVER_IP}:{PORT}/v1/chat/completions"
    payload = {
        "model": "qwen3.8-flash-next-ud-q4_k_xl",
        "messages": [
            {"role": "user", "content": f"Iteration {i}: {prompt}"}
        ],
        "max_tokens": 1200,
        "temperature": 0.7,
        "stream": True
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    iter_start = time.time()
    token_count = 0
    last_telem_time = 0
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and not line_str.startswith("data: [DONE]"):
                    token_count += 1
                
                now = time.time()
                if now - last_telem_time >= 2.0:
                    telem = get_gpu_telemetry()
                    t0, p0, u0 = telem[0]["temp"], telem[0]["pwr"], telem[0]["util"]
                    t1, p1, u1 = telem[1]["temp"], telem[1]["pwr"], telem[1]["util"]
                    elapsed = int(now - start_test_time)
                    print(f"[{elapsed:3d}s] GPU 0: {t0:.1f}°C ({p0:.1f}W, {u0:.0f}%) | GPU 1: {t1:.1f}°C ({p1:.1f}W, {u1:.0f}%) | Tokens: {token_count}")
                    sys.stdout.flush()
                    last_telem_time = now
                    
                    if t0 >= SAFETY_TEMP_LIMIT or t1 >= SAFETY_TEMP_LIMIT:
                        hot_gpu = 0 if t0 >= SAFETY_TEMP_LIMIT else 1
                        hot_temp = max(t0, t1)
                        print(f"\n[NÖDSTOPP TRIGGERAT!] GPU {hot_gpu} nådde {hot_temp:.1f}°C (>= {SAFETY_TEMP_LIMIT}°C)!")
                        stopped_early = True
                        break
        
        if stopped_early:
            break
            
        iter_dur = time.time() - iter_start
        print(f"Körning {i} klar: {token_count} tokens genererade på {iter_dur:.1f}s (~{token_count/iter_dur:.1f} tok/s)")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Fel under generering: {e}")
        break

final_telem = get_gpu_telemetry()
print("\n" + "=" * 70)
if stopped_early:
    print(f"TESTET AVBRÖTS AV SÄKERHETSVAKTEN VID >= {SAFETY_TEMP_LIMIT}°C")
else:
    print("TESTET SLUTFÖRDES FRAMGÅNGSRIKT!")
print(f"Sluttemperaturer:")
print(f"  GPU 0: {final_telem[0]['temp']:.1f}°C (Effekt: {final_telem[0]['pwr']:.1f}W)")
print(f"  GPU 1: {final_telem[1]['temp']:.1f}°C (Effekt: {final_telem[1]['pwr']:.1f}W)")
print("=" * 70)
