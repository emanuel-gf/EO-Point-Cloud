import subprocess
import sys
from loguru import logger
import os
from pathlib import Path

def check_apt_package_installed(package_name, version=None):
    """
    Check if an apt package is installed using dpkg-query.
    
    Args:
        package_name (str): Name of the package to check
        version (str, optional): Specific version to check
    
    Returns:
        bool: True if package is installed, False otherwise
    """
    try:
        if version:
            # Check for specific version
            cmd = ["dpkg-query", "-W", "-f='${Status} ${Version}'", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip().strip("'")
            
            # Check if status contains "install ok installed" and version matches
            if "install ok installed" in output and version in output:
                logger.info(f" {package_name} version {version} is installed")
                return True
            else:
                logger.warning(f" {package_name} version {version} is NOT installed or version mismatch")
                logger.info(f"Current status: {output}")
                return False
        else:
            # Check if package exists (any version)
            cmd = ["dpkg-query", "-W", "-f='${Status}'", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip().strip("'")
            
            if "install ok installed" in output:
                logger.info(f" {package_name} is installed")
                return True
            else:
                logger.warning(f" {package_name} is NOT installed")
                return False
                
    except subprocess.CalledProcessError:
        logger.error(f" {package_name} is NOT installed (package not found)")
        return False
    except Exception as e:
        logger.error(f"Error checking {package_name}: {e}")
        return False

def check_all_apt_requirements():
    """
    Check all apt requirements from your manifest file.
    """
    # Your apt requirements from the manifest
    apt_requirements = [
        {"name": "gdal-bin", "version": "3.6.2+dfsg-1+b2"},
        {"name": "libgdal-dev", "version": "3.6.2+dfsg-1+b2"},
        {"name": "libx11-6", "version": "2:1.8.12-1"},
        {"name": "libx11-dev", "version": "2:1.8.12-1"},
        {"name": "libx11-xcb1", "version": "2:1.8.12-1"},
        {"name": "libxcb1", "version": "1.17.0-2+b1"},
        {"name": "libxrandr2", "version": "2:1.5.4-1+b3"},
        {"name": "libxinerama1", "version": "2:1.1.4-3+b4"},
        {"name": "libxcursor1", "version": "1:1.2.3-1"},
        {"name": "libxi6", "version": "2:1.8.2-1"},
        {"name": "libgl1", "version": "1.7.0-1+b2"},
        {"name": "libglx-mesa0", "version": "25.0.7-2"},
        {"name": "libgl1-mesa-dri", "version": "25.0.7-2"},
        {"name": "libegl1", "version": "1.7.0-1+b2"},
        {"name": "libegl-mesa0", "version": "25.0.7-2"},
        {"name": "xvfb", "version": "2:21.1.16-1.3"}
    ]
    
    logger.info("Checking APT package installations...")
    all_installed = True
    
    for package in apt_requirements:
        package_name = package["name"]
        package_version = package["version"]
        
        is_installed = check_apt_package_installed(package_name, package_version)
        if not is_installed:
            all_installed = False
    
    if all_installed:
        logger.success(" All APT packages are installed correctly!")
    else:
        logger.error(" Some APT packages are missing or have incorrect versions!")
        sys.exit(1)
    
    return all_installed

def check_critical_libraries():
    """
    Check if critical libraries are accessible from Python.
    """
    logger.info("Checking critical library accessibility...")
    
    # Test GDAL
    try:
        from osgeo import gdal
        logger.info(" GDAL is accessible from Python")
    except ImportError as e:
        logger.error(f" GDAL not accessible: {e}")
    
    # Test OpenGL libraries (indirectly through Open3D)
    try:
        import open3d as o3d
        # Try to create a simple visualization (this will test OpenGL dependencies)
        mesh = o3d.geometry.TriangleMesh.create_sphere()
        logger.info(" OpenGL libraries are working (Open3D can create geometry)")
    except Exception as e:
        logger.warning(f"OpenGL/Open3D warning: {e}")
    
    # Test X11 libraries (if in display environment)
    try:
        import os
        if 'DISPLAY' in os.environ:
            # Only test if display is available
            result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(" X11 display system is working")
            else:
                logger.info("X11 display not available (expected in headless environment)")
        else:
            logger.info("No DISPLAY environment variable (headless mode)")
    except Exception as e:
        logger.info(f"X11 check skipped: {e}")

def import_open3d():
    """
    try to import open3d in a headless mode
    """
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
        logger.error(f"Impossible to install")

if __name__ == "__main__":
    # Run the checks
    check_all_apt_requirements()
    check_critical_libraries()
    import_open3d
    
    logger.success("verification completed!")