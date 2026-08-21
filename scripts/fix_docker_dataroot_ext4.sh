#!/usr/bin/env bash
# Fix: Docker data-root on /mnt/e (9p/DrvFs) cannot host containerd mounts
# (mkdir .../dev/shm: no such file or directory).
#
# Solution: sparse ext4 image on E:, loop-mounted at /var/lib/docker.
# Disk space still lives on E:; filesystem semantics are Linux.
#
# Run: wsl -d Ubuntu -- bash /mnt/c/Users/kashy/Desktop/KernelFuse/scripts/fix_docker_dataroot_ext4.sh
set -euo pipefail

IMG="${DOCKER_DATA_IMG:-/mnt/e/Docker/docker-data.img}"
MNT="${DOCKER_DATA_MNT:-/var/lib/docker}"
# Sparse file; grows with use. Override: DOCKER_DATA_SIZE_GB=120
SIZE_GB="${DOCKER_DATA_SIZE_GB:-80}"

if [[ ! -d /mnt/e ]]; then
  echo "ERROR: /mnt/e not available"
  exit 1
fi

echo "=== stop docker ==="
sudo service docker stop 2>/dev/null || sudo systemctl stop docker 2>/dev/null || true
sleep 1

echo "=== create sparse ext4 image (${SIZE_GB}G) at ${IMG} ==="
sudo mkdir -p "$(dirname "$IMG")"
if [[ ! -f "$IMG" ]]; then
  sudo truncate -s "${SIZE_GB}G" "$IMG"
  sudo mkfs.ext4 -F -L docker-data "$IMG"
else
  echo "image exists, reusing: $IMG"
fi

echo "=== mount ${IMG} -> ${MNT} ==="
sudo mkdir -p "$MNT"
# Unmount if something else is there
if mountpoint -q "$MNT" 2>/dev/null; then
  sudo umount "$MNT" || true
fi
sudo mount -o loop "$IMG" "$MNT"
df -Th "$MNT"

echo "=== fstab for WSL reboot persistence ==="
FSTAB_LINE="${IMG} ${MNT} ext4 loop,defaults 0 0"
if ! grep -qF "$IMG" /etc/fstab 2>/dev/null; then
  echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null
  echo "appended to /etc/fstab"
else
  echo "fstab already has $IMG"
fi

echo "=== daemon.json: data-root = ${MNT} ==="
sudo mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
  sudo cp /etc/docker/daemon.json "/etc/docker/daemon.json.bak.$(date +%s)"
fi
# Keep nvidia runtime; point data-root at the ext4 mount.
sudo tee /etc/docker/daemon.json >/dev/null <<EOF
{
  "data-root": "${MNT}",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

# Optional: leave old 9p tree alone (safe to delete later to reclaim E: space)
OLD=/mnt/e/Docker/engine
if [[ -d "$OLD" ]]; then
  echo "note: old 9p data-root still at $OLD — remove after this works:"
  echo "  sudo rm -rf $OLD"
fi

echo "=== start docker ==="
sudo service docker start || sudo systemctl start docker
sleep 2
docker info | grep -E "Docker Root Dir|Storage Driver" || sudo docker info | grep -E "Docker Root Dir|Storage Driver"

echo "=== GPU smoke ==="
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi \
  || sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

echo "Done."
