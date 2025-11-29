# tested on Ubuntu 22.04.3 LTS
# download and install blender 4.1.0 on root/opt directory
set -e

# Load configuration
source "$(dirname "$0")/config.sh"

# wget https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz
# tar -xvf blender-4.1.0-linux-x64.tar.xz
mv blender-4.1.0-linux-x64 "$BLENDER_PATH"
rm blender-4.1.0-linux-x64.tar.xz
ln -sf "$BLENDER_PATH/blender" /usr/bin/blender

# Add project path to Blender's python site-packages
echo "$(dirname "$(readlink -f "$0")")" | tee "$BLENDER_PATH/4.1/python/lib/python3.11/site-packages/clevr.pth"

# download and install blender package
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libglu1-mesa \
  libxi6 \
  libxrender1 \
  libxrandr2 \
  libxcursor1 \
  libxinerama1 \
  libxxf86vm1 \
  libxfixes3 \
  libxkbcommon0 \
  libtbb2 \
  libsm6 \
  libice6 \
  libgl1-mesa-glx

# verify installation
blender --version
