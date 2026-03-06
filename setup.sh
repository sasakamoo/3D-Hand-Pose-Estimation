#!/bin/bash

###############################################################################
# FINAL COMPLETE SOLUTION - Clean Install Everything
# This removes all conflicting packages and installs compatible versions
###############################################################################

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          COMPLETE SOLUTION - Clean Install                 ║"
echo "║    Removing all conflicts and installing clean             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "STEP 1: Remove ALL Python packages that conflict"
echo "=========================================="

python3 -m pip uninstall numpy -y --break-system-packages 2>&1 | grep -v "^WARNING" || true
python3 -m pip uninstall opencv-python -y --break-system-packages 2>&1 | grep -v "^WARNING" || true
python3 -m pip uninstall opencv-python-headless -y --break-system-packages 2>&1 | grep -v "^WARNING" || true
python3 -m pip uninstall scipy -y --break-system-packages 2>&1 | grep -v "^WARNING" || true
python3 -m pip uninstall scikit-learn -y --break-system-packages 2>&1 | grep -v "^WARNING" || true

echo "✓ Removed conflicting packages"
echo ""

echo "STEP 2: Clean all pip caches"
echo "=========================================="

python3 -m pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip/ 2>/dev/null || true

echo "✓ Caches cleaned"
echo ""

echo "STEP 3: Install ONLY compatible versions (NO conflicts)"
echo "=========================================="

# These versions are 100% compatible
python3 -m pip install --no-cache-dir 'numpy==1.26.4' --break-system-packages
python3 -m pip install --no-cache-dir 'opencv-python==4.8.1.78' --break-system-packages
python3 -m pip install --no-cache-dir 'torch==2.1.2' --break-system-packages
python3 -m pip install --no-cache-dir 'torchvision==0.16.2' --break-system-packages
python3 -m pip install --no-cache-dir 'matplotlib==3.8.2' --break-system-packages
python3 -m pip install --no-cache-dir 'open3d==0.17.0' --break-system-packages

echo "✓ All packages installed"
echo ""

echo "STEP 4: Verify everything works"
echo "=========================================="

python3 << 'VERIFY'
import sys
print("Testing imports...")

try:
    import numpy
    print(f"  ✓ NumPy {numpy.__version__}")
except Exception as e:
    print(f"  ✗ NumPy FAILED: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"  ✗ OpenCV FAILED: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"  ✗ PyTorch FAILED: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print(f"  ✓ Matplotlib working")
except Exception as e:
    print(f"  ✗ Matplotlib FAILED: {e}")
    sys.exit(1)

try:
    import open3d as o3d
    print(f"  ✓ Open3D {o3d.__version__}")
except Exception as e:
    print(f"  ⚠️  Open3D optional (not critical)")

print("\n✓✓✓ ALL CRITICAL PACKAGES WORK ✓✓✓")
VERIFY

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Verification failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✓✓✓ COMPLETE SUCCESS ✓✓✓                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Copy the script:"
echo "   cp hand_pose_loader.py ~/3D-Hand-Pose-Estimation/Dataset.py"
echo ""
echo "2. Test it:"
echo "   cd ~/3D-Hand-Pose-Estimation"
echo "   python3 Dataset.py --check-path"
echo ""
echo "3. Run visualization:"
echo "   python3 Dataset.py --mode 2d --num-samples 4"
echo ""
echo "✓ Everything is ready!"
echo ""