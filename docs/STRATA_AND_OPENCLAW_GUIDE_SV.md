# Komplett Reinstallationsmanual & Postmortem för Strata, HP Z8 G4 & OpenClaw

Detta dokument är en komplett, uttömmande steg-för-steg-guide för att från grunden installera om hela stacken: operativsystemoptimeringar, systemd-tjänster, termisk styrning av dual Tesla V100, kompilering av `strata-vision`, modellvikter, MTP-tabeller, Strata-konfiguration samt OpenClaw-kopplingen.

---

## 1. Systemöversikt & Hårdvara

* **Dator:** HP Z8 G4 Workstation
* **CPU:** 2x Intel Xeon Gold 6140 @ 2.30 GHz (36 kärnor, 72 trådar, Dual Socket NUMA)
* **RAM:** 64 GB ECC DDR4-2666 (4x 16GB RDIMM)
* **GPU 0 & 1 (Inference):** 2x NVIDIA Tesla V100 PCIe 32GB (64GB HBM2, Volta `sm_70`)
* **GPU 2 (Display):** NVIDIA GeForce GT 1030 2GB GDDR5 (Isolerad från beräkning)
* **IP-adress:** `192.168.1.240` (Port 8086 för Strata API)

---

## 2. Operativsystem & Kärninställningar (Steg för steg)

### A. Paket och Byggverktyg (APT)
Tesla V100 (Volta) kräver CUDA 12-verktygskedjan (eftersom CUDA 13+ fasar ut Volta `sm_70`):
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git curl jq \
                    nvidia-cuda-toolkit python3-pip python3-venv
```

### B. Skydda `/dev/shm` mot Systemd Session Wipeout (Kritiskt!)
Som standard har Ubuntu/Debian `RemoveIPC=yes` och `Linger=no`. Detta gör att systemd automatiskt rensar delat minne i `/dev/shm` så fort en användare kopplar ifrån SSH, vilket tidigare kraschade Stratas bildcache:
```bash
# 1. Aktivera user linger så att användarens processer och minne aldrig rensas vid SSH-logout
sudo loginctl enable-linger user

# 2. Skapa drop-in för systemd-logind som permanent stänger av IPC/shm-städning
sudo mkdir -p /etc/systemd/logind.conf.d
echo -e "[Login]\nRemoveIPC=no" | sudo tee /etc/systemd/logind.conf.d/no-remove-ipc.conf

# 3. Starta om logind
sudo systemctl restart systemd-logind
```

### C. CPU Energi- och NUMA-profilering (`cpu-eco-profile.service`)
Säkerställer optimal frekvensskalning på båda Xeon-socklarna:
Skapa `/etc/systemd/system/cpu-eco-profile.service`:
```ini
[Unit]
Description=Set Intel CPU Energy Profile for Eco Idle and Dynamic Boost
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for e in /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference; do echo balance_performance > "$e" 2>/dev/null || true; done; for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo powersave > "$g" 2>/dev/null || true; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
Aktivera:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-eco-profile.service
```

---

## 3. Tesla V100 Termisk Profil & Klocklåsning

### A. Den Termiska "Sweet Spoten" (155W / 1380 MHz)
V100-korten är passivt kylda beräkningskort i en workstation.
- Standard 250W eller 190W leder till temperaturer över 75–85°C.
- **155W effektgräns** kombinerat med **låsta basklockor på 1380 MHz** (`nvidia-smi -lgc 1380,1380`) sänker temperaturerna till **42–47°C idle** och **max 58°C under 67k-tokens kontext** med bibehållen 98%+ beräkningsprestanda (1295 tok/s prefill, ~120 tok/s decode).

### B. Watchdog-skript: `/usr/local/bin/v100-thermal-guard.py`
Skriptet sätter klockor, aktiverar Persistence Mode och implementerar stegvis termisk nedtrappning (78°C -> 120W, 84°C -> 100W, 88°C -> clamp).

Skapa `/etc/systemd/system/v100-thermal-guard.service`:
```ini
[Unit]
Description=NVIDIA Tesla V100 155W Power Cap and Thermal Watchdog Daemon
After=multi-user.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /usr/local/bin/v100-thermal-guard.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
Aktivera:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now v100-thermal-guard.service
```

---

## 4. Strata Modell, MTP-tabeller & Bygge av `strata-vision`

### A. Katalogstruktur
```bash
mkdir -p /home/user/Strata-data/models
mkdir -p /home/user/Strata-data/mtp/rt
mkdir -p /home/user/strata/data
```

### B. Modellfiler (Qwen3.8-Flash-Next IQ3_S + Vision)
Hämtas från ISTA-DASLab / HuggingFace:
1. `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54.8 GB) -> `/home/user/Strata-data/models/`
2. `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28.8 GB) -> `/home/user/Strata-data/models/`
3. `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (Vision Projector) -> `/home/user/Strata-data/models/`

*Viktigt:* Den automatiska bakgrundsnedladdaren (`strata-model-downloader.service`) **måste inaktiveras** efter att filerna är nedladdade, annars startar den om var tionde sekund:
```bash
sudo systemctl disable --now strata-model-downloader.service
```

### C. Kompilera `strata-vision` för Volta V100 (SM70)
Eftersom förbyggda binärer saknar `sm_70` måste `strata-vision` byggas från källkod med Ninja och CMake:
```bash
cd /home/user/strata

# Konfigurera CMake för Volta SM70 med Release-optimering
cmake -B build-vision \
      -S tools/vision \
      -G Ninja \
      -DCMAKE_CUDA_ARCHITECTURES="70" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON

# Bygg binären
ninja -C build-vision strata-vision

# Placera binären i Stratas sökväg
mkdir -p engine-cuda12
cp build-vision/bin/strata-vision engine-cuda12/strata-vision
chmod +x engine-cuda12/strata-vision
```

---

## 5. Strata Konfiguration & Systemd Service

### A. Modellkonfiguration: `/home/user/strata/strata-iq3_xxs.json`
```json
{
  "model": "Strata",
  "engine": "strata",
  "args": [
    "-m", "/home/user/Strata-data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--mmproj", "/home/user/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "--prefill", "8192",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--mtp", "/home/user/Strata-data/mtp/rt",
    "--max-context", "131072",
    "--kv", "int8",
    "--remote-expert-opt",
    "--pipeline-windows", "2",
    "--lookup-chain", "2",
    "--pool-workers", "17",
    "--expert-profile-save", "/home/user/strata/data/expert-profile.bin",
    "--expert-profile-save-every", "5",
    "--vision",
    "--vram-reserve-mib", "700",
    "--layer-split", "auto"
  ]
}
```

*Varför dessa flaggor är kritiska:*
* `--pool-workers 17`: Begränsar trådar till fysiska kärnor på den lokala CPU-sockeln (eliminerar UPI-bussmättnad mellan socklarna).
* `--vram-reserve-mib 700`: Reserverar VRAM på GPU 0 så att `strata-vision`-enkodern inte kolliderar med KV-cachen.
* `--kv int8`: Gör att full 131k kontext ryms i 64GB VRAM utan att spilla över till system-RAM.
* `--lookup-chain 2` och `--pipeline-windows 2`: Pipelinar routing och beräkning av MoE-experter.
* `--spec 4 --spec-min-p 0.5`: Spekulativ avkodning via MTP (ökar avkodningshastighet med ~40%).

### B. Strata Systemd Enhet: `/etc/systemd/system/strata-server.service`
```ini
[Unit]
Description=Strata AI Server (Dual Tesla V100 64GB)
After=network-online.target v100-thermal-guard.service
Wants=network-online.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/strata
ExecStart=/home/user/strata/.venv/bin/python /home/user/strata/serve/server.py --engine strata --config /home/user/strata/strata-iq3_xxs.json --port 8086
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
Environment="CUDA_VISIBLE_DEVICES=0,1"

[Install]
WantedBy=multi-user.target
```
Aktivera och starta:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now strata-server.service
```

---

## 6. Övervakning & Dashboard Tjänster

1. **HP Live Thermal Web Dashboard (`hp-thermal-dashboard.service`):**
   * Tjänst: `/etc/systemd/system/hp-thermal-dashboard.service`
   * Skript: `/usr/local/bin/hp-thermal-monitor.py`
   * Övervakar temperaturer, fläktar och RAPL-effektförbrukning på CPU och GPU:er.
2. **Eco Standby Daemon (`eco-standby.service`):**
   * Tjänst: `/etc/systemd/system/eco-standby.service`
   * Skript: `/usr/local/bin/eco-standby-daemon.py`
   * Sänker systemets effektförbrukning vid längre inaktivitet och väcker systemet omedelbart vid anrop.

---

## 7. OpenClaw Konfiguration & Integration

När OpenClaw kommunicerar med servern:
* **API Base URL:** `http://192.168.1.240:8086/v1`
* **Modellnamn:** `Strata` (eller matchande modellsträng i OpenAI-klienten)
* **Multimodal format:** Skicka skärmdumpar som `data:image/png;base64,...` eller WebP.
* **Reasoning Budget:**
  - Sätt antingen `reasoning_budget_tokens: 4096` (så att modellen inte bryts i förtid under bildinspektion) eller skicka `reasoning_effort: "low"` för enkla visuella verifieringar.
* **Stream Resilience:**
  - Om en ström klipps, använd den sparade konversationen och skicka en återupptagningsfråga istället för att terminera agenten.

---

## 8. Verifieringschecklista efter ominstallation

Kör följande snabbtest för att verifiera att hela kedjan fungerar:

1. **Kontrollera GPU-klockor och effektgräns:**
   ```bash
   nvidia-smi --query-gpu=index,power.limit,clocks.current.graphics,temperature.gpu --format=csv,noheader
   # Förväntat: 170.00 W, 1380 MHz, <50°C
   ```
2. **Kontrollera Linger och RemoveIPC:**
   ```bash
   loginctl show-user user | grep Linger
   # Förväntat: Linger=yes
   ```
3. **Kör ett multimodalt resoneringstest mot Strata:**
   ```bash
   curl -X POST http://127.0.0.1:8086/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Strata",
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "text", "text": "Describe this image."},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}}
           ]
         }
       ],
       "reasoning_budget_tokens": 15,
       "stream": false
     }'
   ```
   *Förväntat resultat:* Status 200 OK med både `reasoning_content` och avslutande text utan några fel i `journalctl -u strata-server.service`.
