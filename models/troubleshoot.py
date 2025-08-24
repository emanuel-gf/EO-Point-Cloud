#!/usr/bin/env python3
"""
Main script to build Open3D from source using Docker and set up headless rendering.
This script handles the complete Open3D installation process within the orchestrator.
"""

import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Execute shell command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        if check:
            raise
        return e

def setup_open3d_from_docker():
    """Build Open3D using Docker and install the wheel."""
    
    # Set up directories
    work_dir = Path("/tmp/open3d_build")
    work_dir.mkdir(exist_ok=True)
    
    print("=== Cloning Open3D repository ===")
    if not (work_dir / "Open3D").exists():
        run_command(f"git clone https://github.com/isl-org/Open3D", cwd=work_dir)
    
    open3d_dir = work_dir / "Open3D"
    docker_dir = open3d_dir / "docker"
    
    # Determine Python version for build configuration
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Choose appropriate build configuration
    # Available options: cuda_wheel_py38_dev, openblas-amd64-py310, etc.
    if python_version == "3.8":
        build_config = "cuda_wheel_py38_dev"
    elif python_version == "3.10":
        build_config = "openblas-amd64-py310" 
    elif python_version == "3.11":
        # Use closest available or modify for 3.11
        build_config = "openblas-amd64-py310"  # Fallback
    else:
        build_config = "openblas-amd64-py310"  # Default fallback
    
    print(f"=== Building Open3D with configuration: {build_config} ===")
    print("This may take 30-60 minutes depending on your system...")
    
    # Make the build script executable
    build_script = docker_dir / "docker_build.sh"
    run_command(f"chmod +x {build_script}")
    
    # Run the Docker build
    try:
        run_command(f"./docker_build.sh {build_config}", cwd=docker_dir)
    except subprocess.CalledProcessError:
        print("Docker build failed. Trying alternative approach...")
        # Try without CUDA if that was the issue
        if "cuda" in build_config:
            build_config = "openblas-amd64-py310"
            print(f"Retrying with: {build_config}")
            run_command(f"./docker_build.sh {build_config}", cwd=docker_dir)
        else:
            raise
    
    # Find the generated wheel file
    wheel_pattern = docker_dir / "output" / "*.whl"
    wheel_files = glob.glob(str(wheel_pattern))
    
    if not wheel_files:
        # Check alternative output locations
        wheel_pattern_alt = open3d_dir / "build" / "lib" / "*.whl"
        wheel_files = glob.glob(str(wheel_pattern_alt))
    
    if not wheel_files:
        raise FileNotFoundError("No wheel file found after build!")
    
    # Install the wheel
    wheel_file = wheel_files[0]  # Take the first/latest wheel
    print(f"=== Installing Open3D wheel: {wheel_file} ===")
    run_command(f"pip install {wheel_file}")
    
    print("=== Open3D installation completed successfully! ===")

def setup_headless_rendering():
    """Configure environment for headless rendering."""
    print("=== Setting up headless rendering environment ===")
    
    # Set environment variables for headless rendering
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['DISPLAY'] = ':99'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    
    # Start Xvfb (X Virtual Framebuffer) for headless X11
    print("Starting Xvfb for headless X11...")
    try:
        # Kill any existing Xvfb processes
        run_command("pkill Xvfb", check=False)
        # Start Xvfb
        subprocess.Popen([
            'Xvfb', ':99', 
            '-screen', '0', '1024x768x24', 
            '-ac', '+extension', 'GLX', '+render', '-noreset'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a moment for Xvfb to start
        import time
        time.sleep(2)
        print("Xvfb started successfully")
    except Exception as e:
        print(f"Warning: Could not start Xvfb: {e}")
    
    # Create a simple test script to verify installation
    test_script = """
import os
import sys

# Set environment variables before importing open3d
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['DISPLAY'] = ':99'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

try:
    import open3d as o3d
    import numpy as np
    
    print("Open3D version:", o3d.__version__)
    print("CUDA available:", o3d.core.cuda.is_available())
    
    # Create a simple test scene
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    
    print("Created test mesh successfully")
    
    # Try headless rendering using render_to_image
    try:
        # Use render_to_image method instead of Visualizer for better headless support
        import open3d.visualization.rendering as rendering
        
        render = rendering.OffscreenRenderer(640, 480)
        render.scene.add_geometry("mesh", mesh, rendering.MaterialRecord())
        img = render.render_to_image()
        
        print("Headless rendering test successful!")
        print("Captured image shape:", np.asarray(img).shape)
        
        # Save test image
        o3d.io.write_image("/tmp/test_render.png", img)
        print("Test image saved to /tmp/test_render.png")
        
    except Exception as render_error:
        print(f"Offscreen rendering failed: {render_error}")
        print("Trying alternative method...")
        
        # Fallback to CPU-only rendering
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=640, height=480)
            vis.add_geometry(mesh)
            vis.update_renderer()
            
            # Try to capture screen
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            
            print("Alternative headless rendering successful!")
            print("Captured image shape:", np.asarray(image).shape)
            
        except Exception as vis_error:
            print(f"Visualizer method also failed: {vis_error}")
            print("Open3D is installed but may require GPU access for rendering")
            sys.exit(1)

except ImportError as e:
    print(f"Failed to import Open3D: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
"""
    
    # Save and run test
    test_file = Path("/tmp/test_open3d.py")
    test_file.write_text(test_script)
    
    try:
        run_command(f"python {test_file}")
        print("=== Headless rendering setup verified! ===")
    except subprocess.CalledProcessError as e:
        print("Warning: Headless rendering test failed, but Open3D is installed")
        print("You may need to configure GPU access or run in CPU-only mode")
        print("Try using OffscreenRenderer for headless rendering in your application")

def fallback_pip_install():
    """Fallback to PyPI installation if Docker build fails."""
    print("=== Falling back to PyPI installation ===")
    run_command("pip install open3d==0.19.0")
    setup_headless_rendering()

def main():
    """Main entry point."""
    print("=== Open3D Setup Starting ===")
    
    # Check if Docker is available
    try:
        run_command("docker --version")
        docker_available = True
    except subprocess.CalledProcessError:
        print("Docker not available, falling back to PyPI installation")
        docker_available = False
    
    try:
        if docker_available:
            setup_open3d_from_docker()
        else:
            fallback_pip_install()
        
        setup_headless_rendering()
        
        print("=== Setup completed successfully! ===")
        print("You can now use Open3D for headless rendering.")
        print("Remember to set EGL_PLATFORM=surfaceless in your environment.")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        print("Attempting fallback installation...")
        try:
            fallback_pip_install()
            print("Fallback installation successful!")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            sys.exit(1)

if __name__ == "__main__":
    main()