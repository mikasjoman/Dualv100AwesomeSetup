#!/usr/bin/env bash
# ==============================================================================
# Dualv100AwesomeSetup - Automated OS, Thermal & Systemd Optimization Installer
# Target Hardware: HP Z8 G4 Workstation (Dual Xeon LGA3647 + Dual Tesla V100 32GB)
# ==============================================================================
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run this installer as root: sudo ./install.sh"
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-user}"

echo "======================================================================"
echo " Applying Dualv100AwesomeSetup Optimizations (Target User: $TARGET_USER)"
echo "======================================================================"

# 1. Prevent systemd session wipeout of /dev/shm (critical for vision & IPC caches)
echo "[1/5] Protecting /dev/shm from systemd-logind session cleanup..."
loginctl enable-linger "$TARGET_USER" || true
mkdir -p /etc/systemd/logind.conf.d
cat << 'EOF' > /etc/systemd/logind.conf.d/no-remove-ipc.conf
[Login]
RemoveIPC=no
EOF

# 2. Install Thermal Guard & Live Telemetry Dashboard daemons
echo "[2/5] Installing V100 Thermal Guard & Live Web Telemetry Daemons..."
install -m 755 "$REPO_DIR/daemons/v100-thermal-guard.py" /usr/local/bin/v100-thermal-guard.py
install -m 755 "$REPO_DIR/daemons/hp-thermal-monitor.py" /usr/local/bin/hp-thermal-monitor.py

if [[ ! -f /etc/v100-power-cap ]]; then
  echo "150" > /etc/v100-power-cap
fi

# 3. Install Systemd Units
echo "[3/5] Installing systemd service files..."
install -m 644 "$REPO_DIR/systemd/cpu-eco-profile.service" /etc/systemd/system/cpu-eco-profile.service
install -m 644 "$REPO_DIR/systemd/v100-thermal-guard.service" /etc/systemd/system/v100-thermal-guard.service
install -m 644 "$REPO_DIR/systemd/hp-thermal-dashboard.service" /etc/systemd/system/hp-thermal-dashboard.service

# 4. Reload and enable system services
echo "[4/5] Enabling and starting CPU Eco Profile, V100 Thermal Guard & Dashboard..."
systemctl daemon-reload
systemctl enable --now cpu-eco-profile.service
systemctl enable --now v100-thermal-guard.service
systemctl enable --now hp-thermal-dashboard.service

# 5. Summary
echo "[5/5] Verifying GPU state..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,power.limit,clocks.current.graphics,temperature.gpu --format=csv
fi

echo "======================================================================"
echo " Done! Live Thermal Dashboard is running on port 9000."
echo " Optional: Copy systemd/strata-server.service to /etc/systemd/system/"
echo " and configs/strata-ud-q4_k_xl.json to your Strata directory."
echo "======================================================================"
