#!/usr/bin/env bash
# Install Docker Engine + NVIDIA Container Toolkit inside WSL2 Ubuntu.
# Space on E: via an ext4 loop image (NOT raw /mnt/e — 9p/DrvFs breaks containerd).
#
# Run: wsl -d Ubuntu -- bash /mnt/c/Users/kashy/Desktop/KernelFuse/scripts/install_docker_wsl_e.sh
# If Engine is already installed and GPU smoke failed with /dev/shm, run instead:
#   scripts/fix_docker_dataroot_ext4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Docker Engine install ==="
if [[ ! -d /mnt/e ]]; then
  echo "ERROR: /mnt/e not mounted — is the E: drive available to WSL?"
  exit 1
fi

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
  sudo chmod a+r /etc/apt/keyrings/docker.asc
fi
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker || true

sudo usermod -aG docker "${SUDO_USER:-$USER}" || true

# Put Docker on ext4 loop on E: (fixes /dev/shm on 9p)
bash "${SCRIPT_DIR}/fix_docker_dataroot_ext4.sh"

echo "Done. Open a new WSL shell so the docker group applies."
