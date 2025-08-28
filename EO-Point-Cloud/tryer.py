#!/usr/bin/env python3
"""
Your existing tryer.py with runtime Open3D setup
"""

import os
import sys
import subprocess
from pathlib import Path

# RUNTIME OPEN3D SETUP - ADD THIS AT THE TOP
def setup_open3d_deps():
    """Quick runtime setup for Open3D dependencies."""
    
    # Set environment variables
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-xvfb'
    
    Path('/tmp/runtime-xvfb').mkdir(exist_ok=True, mode=0o700)
    
    # Quick check if Open3D works
    try:
        import open3d as o3d
        print(f"✅ Open3D {o3d.__version__} ready")
        return True
    except ImportError:
        print("Installing Open3D dependencies...")
    
    # Install critical dependencies
    deps = "libegl1-mesa libgl1-mesa-glx libgomp1 libx11-6 xvfb"
    try:
        subprocess.run("apt-get update", shell=True, check=False, capture_output=True)
        subprocess.run(f"apt-get install -y {deps}", shell=True, check=False, capture_output=True)
        subprocess.run("pip install open3d==0.19.0", shell=True, check=False, capture_output=True)
        
        # Test again
        import open3d as o3d
        print(f"✅ Open3D {o3d.__version__} installed at runtime")
        return True
    except Exception as e:
        print(f"⚠️  Runtime setup had issues: {e}")
        return False

# Run setup immediately
if not setup_open3d_deps():
    print("❌ Could not setup Open3D - proceeding anyway...")

# NOW your original code continues...
import datetime
import time

# Set environment variables early (keep your existing ones)
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

from loguru import logger
# ... rest of your existing imports and code ...