# Step-by-Step Guide: Building a Custom Docker Image with Open3D for Headless Environments

Based on our conversation, here's the complete reproducible process we discovered to solve Open3D headless rendering issues:

## Problem We Solved

- Open3D requires X11 and OpenGL libraries even for headless operation
- `pip install open3d` fails with missing system dependencies
- EGL context creation fails in containerized environments
- Orchestrators often don't support custom base images

## Our Solution: Pre-built Docker Image Approach

### Step 1: Identify Your Current Orchestrator Image

```bash
# Check running containers to find your orchestrator's image
docker ps -a

# Example output:
# CONTAINER ID   IMAGE                                                                                            
# 6bacfb00bf59   nr71n11k.c1.gra9.container-registry.ovh.net/eo-point-cloud-generator/point_cloud_generator:dev
```

### Step 2: Start and Access the Existing Container

```bash
# Start the container (if stopped)
docker start 6bacfb00bf59

# Access the container
docker exec -it 6bacfb00bf59 /bin/bash
```

### Step 3: Install System Dependencies Inside the Container

```bash
# Inside the container, update package lists
apt-get update

# Install all required Open3D system dependencies
apt-get install -y \
    libegl1-mesa \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    xvfb \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for headless rendering
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export XDG_RUNTIME_DIR=/tmp/runtime-xvfb
export MESA_LOADER_DRIVER_OVERRIDE=softpipe

# Create runtime directory
mkdir -p /tmp/runtime-xvfb && chmod 700 /tmp/runtime-xvfb
```

### Step 4: Install Open3D

```bash
# Install Open3D via pip
pip install open3d==0.19.0

# Test the installation
python -c "import open3d as o3d; print(f'Open3D {o3d.__version__} installed successfully')"
```

### Step 5: Exit and Commit the Modified Container

```bash
# Exit the container
exit

# Commit the changes to create a new image
docker commit 6bacfb00bf59 your-registry/point_cloud_generator:open3d-v1

# Verify the new image was created
docker images | grep point_cloud_generator
```

### Step 6: Test the Modified Image

```bash
# Test basic Open3D functionality
docker run --rm your-registry/point_cloud_generator:open3d-v1 \
    python -c "import open3d as o3d; print('Open3D version:', o3d.__version__)"

# Test with environment variables
docker run --rm \
    -e EGL_PLATFORM=surfaceless \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    your-registry/point_cloud_generator:open3d-v1 \
    python -c "
import os
import open3d as o3d

# Test basic operations
mesh = o3d.geometry.TriangleMesh.create_box()
pcd = mesh.sample_points_uniformly(1000)
print(f'SUCCESS: Created mesh with {len(mesh.vertices)} vertices and point cloud with {len(pcd.points)} points')
"
```

### Step 7: Replace the Original Tag (For Orchestrator Compatibility)

```bash
# Tag the modified image with the original tag name
docker tag your-registry/point_cloud_generator:open3d-v1 \
           nr71n11k.c1.gra9.container-registry.ovh.net/eo-point-cloud-generator/point_cloud_generator:dev

# Verify both tags point to the same image
docker images | grep point_cloud_generator
```

### Step 8: Create Optimized Application Code

Create a simplified `main.py` that leverages the pre-built environment:

```python
import os
import sys
from pathlib import Path

# Set environment variables early
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-xvfb'

# Ensure runtime directory exists
Path('/tmp/runtime-xvfb').mkdir(exist_ok=True)

def test_open3d():
    """Test Open3D functionality in pre-built environment."""
    try:
        import open3d as o3d
        print(f"Open3D {o3d.__version__} ready")
        
        # Test basic operations
        mesh = o3d.geometry.TriangleMesh.create_box()
        print(f" Geometry operations work")
        
        # Test file I/O
        o3d.io.write_triangle_mesh("/tmp/test.ply", mesh)
        print(f" File I/O works")
        
        return True
    except Exception as e:
        print(f" Open3D test failed: {e}")
        return False

def main():
    if not test_open3d():
        sys.exit(1)
    
    # Your application logic here
    print(" Open3D is ready!")

if __name__ == "__main__":
    main()
```

### Step 9: Simplify Your Manifest

Update your `manifest.json` to remove Open3D from pip requirements:

```json
{
    "parameters": {
        "pythonVersion": "3.12.2",
        "aptRequirements": [
            {
                "name": "gdal-bin",
                "version": "3.6.2+dfsg-1+b2"
            },
            {
                "name": "libgdal-dev", 
                "version": "3.6.2+dfsg-1+b2"
            }
        ],
        "pipRequirements": [
            // Your other dependencies - NO open3d here
            { "name": "numpy", "version": "2.0" },
            { "name": "matplotlib", "version": "3.10.1" }
            // etc...
        ]
    }
}
```

## Alternative: Dockerfile Approach (If You Have Registry Access)

If you can push to a registry, create this `Dockerfile`:

```dockerfile
FROM nr71n11k.c1.gra9.container-registry.ovh.net/eo-point-cloud-generator/point_cloud_generator:dev

# Switch to root to install packages
USER root

# Install Open3D system dependencies
RUN apt-get update && apt-get install -y \
    libegl1-mesa libgl1-mesa-glx libglu1-mesa \
    libgomp1 libx11-6 libxext6 libxrender1 \
    libxtst6 libxi6 libxrandr2 xvfb mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up environment for headless rendering
ENV EGL_PLATFORM=surfaceless
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-xvfb
ENV MESA_LOADER_DRIVER_OVERRIDE=softpipe

# Create runtime directory
RUN mkdir -p /tmp/runtime-xvfb && chmod 700 /tmp/runtime-xvfb

# Install Open3D
RUN pip install --no-cache-dir open3d==0.19.0

# Test installation
RUN python -c "import open3d as o3d; print(f'Open3D {o3d.__version__} ready!')"

# Switch back to original user if needed
# USER $ORIGINAL_USER
```

Build and push:
```bash
docker build -f Dockerfile -t your-registry/point_cloud_generator:open3d-v1 .
docker push your-registry/point_cloud_generator:open3d-v1
```

## Key Insights We Discovered

1. **`open3d-cpu` isn't always enough** - Sometimes you need the full Open3D with system libraries
2. **X11 libraries are required even for headless** - Open3D imports them even in CPU mode
3. **Environment variables are critical** - `EGL_PLATFORM=surfaceless` and `LIBGL_ALWAYS_SOFTWARE=1`
4. **Runtime directory needed** - `/tmp/runtime-xvfb` must exist
5. **Docker image replacement works** - You can override orchestrator images locally
6. **Pre-built approach is most reliable** - Avoids runtime installation issues

## Verification Commands

```bash
# Check if image has Open3D
docker run --rm your-image python -c "import open3d; print(' Works')"

# Check system libraries
docker run --rm your-image ldconfig -p | grep -E "(libX11|libGL|libEGL|libgomp)"

# Full integration test  
docker run --rm your-image python -c "
import os
os.environ['EGL_PLATFORM'] = 'surfaceless'
import open3d as o3d
mesh = o3d.geometry.TriangleMesh.create_box()
o3d.io.write_triangle_mesh('/tmp/test.ply', mesh)
print(' Full Open3D pipeline works!')
"
```

## Troubleshooting Common Issues

### Issue 1: "libX11.so.6: cannot open shared object file"
**Solution**: Install X11 libraries as shown in Step 3

### Issue 2: "eglCreateContext failed with EGL_BAD_MATCH"
**Solution**: Set environment variables and ensure Mesa libraries are installed

### Issue 3: "Segmentation fault (core dumped)"
**Solution**: 
- Set `LIBGL_ALWAYS_SOFTWARE=1`
- Create `/tmp/runtime-xvfb` directory
- Use `open3d-cpu` instead of full `open3d` if rendering isn't needed

### Issue 4: Container starts but Open3D import fails
**Solution**: Check if all system dependencies are installed:
```bash
docker run --rm your-image apt list --installed | grep -E "(mesa|egl|gl|x11)"
```

### Issue 5: "No such image" when tagging
**Solution**: Use the correct container ID or image ID:
```bash
docker images  # Find the correct IMAGE ID
docker tag IMAGE_ID new-tag-name
```

## Benefits of This Approach

-  **Faster deployment** - No runtime installation of Open3D
-  **Reliable environment** - Pre-tested and consistent
-  **Full Open3D functionality** - All features available
-  **Orchestrator compatible** - Works with existing deployment pipelines
-  **Easy maintenance** - Single source of truth for dependencies

## Summary

This approach gives you a **production-ready, reliable Open3D environment** that works consistently in orchestrated containers! The key is pre-building the Docker image with all necessary system dependencies rather than trying to install them at runtime.

The solution we developed solves the fundamental issue that Open3D has complex system-level dependencies that are difficult to install reliably in containerized environments, especially headless ones.