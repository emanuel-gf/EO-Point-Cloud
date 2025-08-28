import os
import sys
import subprocess
import platform
import ctypes
import ctypes.util
from pathlib import Path
from loguru import logger
import importlib.util
import json

def run_command(cmd, description, timeout=10):
    """Helper function to run system commands safely"""
    try:
        logger.info(f"{description}:")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            logger.info(f"  Output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"  Error: {result.stderr.strip()}")
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        logger.error(f"  Command failed: {e}")
        return None, str(e)

def check_package_versions():
    """Check versions of all installed packages"""
    logger.info("\n9. DETAILED PACKAGE VERSION ANALYSIS")
    logger.info("-" * 40)
    
    # Get all installed packages with versions
    try:
        stdout, stderr = run_command("dpkg -l | grep '^ii'", "All installed packages")
        if stdout:
            packages = {}
            for line in stdout.split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    packages[parts[1]] = parts[2]
            
            # Check specific packages we care about
            important_packages = [
                'libx11-6', 'libx11-dev', 'libx11-xcb1', 'libxcb1',
                'libxrandr2', 'libxinerama1', 'libxcursor1', 'libxi6',
                'libgl1', 'libglx-mesa0', 'libgl1-mesa-dri', 'libgl1-mesa-glx',
                'libegl1', 'libegl-mesa0', 'libegl1-mesa',
                'gdal-bin', 'libgdal-dev', 'libgdal32', 'libgdal32t64',
                'xvfb', 'xserver-xorg-core'
            ]
            
            logger.info("Package versions for important libraries:")
            for pkg in important_packages:
                version = packages.get(pkg, "NOT INSTALLED")
                if version != "NOT INSTALLED":
                    logger.success(f"  ✓ {pkg}: {version}")
                else:
                    logger.error(f"  ✗ {pkg}: {version}")
    except Exception as e:
        logger.error(f"Failed to get package versions: {e}")

def check_gdal_comprehensive():
    """Comprehensive GDAL checking"""
    logger.info("\n10. COMPREHENSIVE GDAL ANALYSIS")
    logger.info("-" * 40)
    
    # Check GDAL binaries
    gdal_binaries = ['gdalinfo', 'gdal-config', 'ogr2ogr', 'gdalwarp']
    for binary in gdal_binaries:
        stdout, stderr = run_command(f"which {binary}", f"Location of {binary}")
        if stdout:
            # Get version if binary exists
            run_command(f"{binary} --version", f"{binary} version")
    
    # Check GDAL libraries
    gdal_libs = [
        'libgdal.so', 'libgdal.so.32', 'libgdal.so.33',
        'libproj.so', 'libgeos.so', 'libgeotiff.so'
    ]
    
    logger.info("GDAL library availability:")
    for lib in gdal_libs:
        try:
            lib_name = lib.replace('.so', '').replace('.32', '').replace('.33', '')
            lib_path = ctypes.util.find_library(lib_name)
            if lib_path:
                logger.success(f"  ✓ {lib} found at: {lib_path}")
            else:
                logger.error(f"  ✗ {lib} NOT FOUND")
        except Exception as e:
            logger.error(f"  ✗ Error checking {lib}: {e}")
    
    # Check GDAL Python binding
    try:
        from osgeo import gdal, ogr, osr
        logger.success("✓ GDAL Python bindings imported successfully")
        logger.info(f"  GDAL version: {gdal.VersionInfo()}")
        logger.info(f"  GDAL release name: {gdal.VersionInfo('RELEASE_NAME')}")
        
        # Test basic GDAL functionality
        try:
            driver_count = gdal.GetDriverCount()
            logger.info(f"  Available GDAL drivers: {driver_count}")
            
            # List some common drivers
            common_drivers = ['GTiff', 'HDF4', 'HDF5', 'netCDF', 'JPEG', 'PNG']
            available_drivers = []
            for driver_name in common_drivers:
                driver = gdal.GetDriverByName(driver_name)
                if driver:
                    available_drivers.append(driver_name)
            logger.info(f"  Common drivers available: {available_drivers}")
            
        except Exception as gdal_test_error:
            logger.error(f"  GDAL functionality test failed: {gdal_test_error}")
            
    except ImportError as gdal_import_error:
        logger.error(f"✗ GDAL Python bindings import failed: {gdal_import_error}")
    except Exception as gdal_error:
        logger.error(f"✗ GDAL check failed: {gdal_error}")
    
    # Check GDAL environment variables
    gdal_env_vars = ['GDAL_DATA', 'PROJ_LIB', 'GDAL_DRIVER_PATH']
    logger.info("GDAL environment variables:")
    for var in gdal_env_vars:
        value = os.environ.get(var, 'NOT SET')
        logger.info(f"  {var}: {value}")
        if value != 'NOT SET' and os.path.exists(value):
            try:
                files = list(Path(value).glob('*'))[:10]  # First 10 files
                logger.info(f"    Contents (first 10): {[f.name for f in files]}")
            except Exception:
                pass

def check_library_dependencies():
    """Check library dependencies using ldd"""
    logger.info("\n11. LIBRARY DEPENDENCY ANALYSIS")
    logger.info("-" * 40)
    
    # Check Python executable dependencies
    run_command("ldd $(which python3)", "Python3 dependencies")
    
    # Check if we can find specific library files and their dependencies
    important_libs = [
        '/usr/lib/x86_64-linux-gnu/libX11.so.6',
        '/usr/lib/x86_64-linux-gnu/libGL.so.1',
        '/usr/lib/x86_64-linux-gnu/libgdal.so'
    ]
    
    for lib_path in important_libs:
        if os.path.exists(lib_path):
            run_command(f"ldd {lib_path}", f"Dependencies of {lib_path}")
        else:
            logger.warning(f"Library not found: {lib_path}")

def generate_debug_summary():
    """Generate a summary report"""
    logger.info("\n12. DEBUG SUMMARY REPORT")
    logger.info("-" * 40)
    
    summary = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "architecture": platform.architecture()[0],
        "environment_variables": {},
        "critical_libraries": {},
        "package_status": {},
        "recommendations": []
    }
    
    # Environment variables
    env_vars = ['DISPLAY', 'LD_LIBRARY_PATH', 'GDAL_DATA', 'PROJ_LIB']
    for var in env_vars:
        summary["environment_variables"][var] = os.environ.get(var, 'NOT SET')
    
    # Critical libraries check
    critical_libs = ['X11', 'GL', 'gdal']
    for lib in critical_libs:
        lib_path = ctypes.util.find_library(lib)
        summary["critical_libraries"][lib] = lib_path if lib_path else "NOT FOUND"
    
    # Generate recommendations
    if summary["critical_libraries"]["X11"] == "NOT FOUND":
        summary["recommendations"].append("Install X11 libraries: libx11-6, libx11-dev")
    
    if summary["critical_libraries"]["GL"] == "NOT FOUND":
        summary["recommendations"].append("Install OpenGL libraries: libgl1, libglx-mesa0")
    
    if summary["critical_libraries"]["gdal"] == "NOT FOUND":
        summary["recommendations"].append("Install GDAL libraries: gdal-bin, libgdal-dev")
    
    if summary["environment_variables"]["DISPLAY"] == "NOT SET":
        summary["recommendations"].append("Set DISPLAY environment variable for X11 applications")
    
    logger.info("Summary Report:")
    logger.info(json.dumps(summary, indent=2))
    
    return summary

def debug_container_environment():
    """Comprehensive debugging of container environment for Open3D and GDAL issues"""
    
    logger.info("=" * 80)
    logger.info("STARTING ENHANCED CONTAINER ENVIRONMENT DEBUG")
    logger.info("=" * 80)
    
    # 1. Basic System Information
    logger.info("1. SYSTEM INFORMATION")
    logger.info("-" * 40)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # 2. Environment Variables
    logger.info("\n2. RELEVANT ENVIRONMENT VARIABLES")
    logger.info("-" * 40)
    env_vars_to_check = [
        'DISPLAY', 'EGL_PLATFORM', 'PYOPENGL_PLATFORM', 
        'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH',
        'XDG_RUNTIME_DIR', 'WAYLAND_DISPLAY',
        'GDAL_DATA', 'PROJ_LIB', 'GDAL_DRIVER_PATH'
    ]
    
    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.info(f"{var}: {value}")
    
    # 3. Check X11 Libraries
    logger.info("\n3. X11 LIBRARY AVAILABILITY")
    logger.info("-" * 40)
    
    x11_libs = [
        'libX11.so.6', 'libX11.so', 'libXrandr.so.2', 'libXrandr.so',
        'libXinerama.so.1', 'libXinerama.so', 'libXcursor.so.1', 'libXcursor.so',
        'libXi.so.6', 'libXi.so', 'libGL.so.1', 'libGL.so', 'libEGL.so.1', 'libEGL.so'
    ]
    
    for lib in x11_libs:
        try:
            lib_name = lib.replace('.so', '').replace('.1', '').replace('.2', '').replace('.6', '')
            lib_path = ctypes.util.find_library(lib_name)
            if lib_path:
                logger.success(f"✓ {lib} found at: {lib_path}")
            else:
                logger.error(f"✗ {lib} NOT FOUND")
        except Exception as e:
            logger.error(f"✗ Error checking {lib}: {e}")
    
    # 4. Check library paths
    logger.info("\n4. LIBRARY SEARCH PATHS")
    logger.info("-" * 40)
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        for path in ld_library_path.split(':'):
            if path and os.path.exists(path):
                logger.info(f"✓ Library path exists: {path}")
                try:
                    files = list(Path(path).glob('libX11*'))
                    if files:
                        logger.info(f"  X11 files found: {[f.name for f in files]}")
                except Exception as e:
                    logger.warning(f"  Could not list files in {path}: {e}")
            elif path:
                logger.warning(f"✗ Library path does not exist: {path}")
    else:
        logger.warning("LD_LIBRARY_PATH is not set")
    
    # Standard library locations
    standard_lib_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
        '/usr/lib',
        '/lib/x86_64-linux-gnu',
        '/lib64',
        '/lib'
    ]
    
    for path in standard_lib_paths:
        if os.path.exists(path):
            logger.info(f"✓ Standard lib path exists: {path}")
            try:
                x11_files = list(Path(path).glob('libX11*'))
                if x11_files:
                    logger.info(f"  X11 files: {[f.name for f in x11_files[:5]]}")  # Show first 5
                gl_files = list(Path(path).glob('libGL*'))
                if gl_files:
                    logger.info(f"  GL files: {[f.name for f in gl_files[:5]]}")  # Show first 5
                gdal_files = list(Path(path).glob('libgdal*'))
                if gdal_files:
                    logger.info(f"  GDAL files: {[f.name for f in gdal_files[:5]]}")  # Show first 5
            except Exception as e:
                logger.warning(f"  Could not scan {path}: {e}")
        else:
            logger.warning(f"✗ Standard lib path missing: {path}")
    
    # 5. Check installed packages
    logger.info("\n5. INSTALLED PACKAGES CHECK")
    logger.info("-" * 40)
    
    packages_to_check = [
        'libx11-6', 'libx11-dev', 'libgl1-mesa-glx', 'libegl1-mesa',
        'libxrandr2', 'libxinerama1', 'libxcursor1', 'libxi6',
        'gdal-bin', 'libgdal-dev', 'libgdal32', 'libgdal32t64'
    ]
    
    for package in packages_to_check:
        try:
            result = subprocess.run(['dpkg', '-l', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    status_line = lines[-1]
                    if status_line.startswith('ii'):
                        parts = status_line.split()
                        version = parts[2] if len(parts) > 2 else "unknown"
                        logger.success(f"✓ {package} is installed (version: {version})")
                    else:
                        logger.warning(f"? {package} status unclear: {status_line}")
                else:
                    logger.warning(f"? {package} status unclear")
            else:
                logger.error(f"✗ {package} is NOT installed")
        except Exception as e:
            logger.error(f"✗ Error checking {package}: {e}")
    
    # 6. Check Python packages
    logger.info("\n6. PYTHON PACKAGES")
    logger.info("-" * 40)
    
    python_packages = ['open3d', 'numpy', 'matplotlib', 'rasterio', 'geopandas']
    for package in python_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec:
                logger.success(f"✓ {package} is available")
                if spec.origin:
                    logger.info(f"  Location: {spec.origin}")
                # Try to get version
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    logger.info(f"  Version: {version}")
                except:
                    pass
            else:
                logger.warning(f"? {package} not found in Python path")
        except Exception as e:
            logger.error(f"✗ Error checking {package}: {e}")
    
    # 7. Try to load Open3D and capture detailed error
    logger.info("\n7. OPEN3D IMPORT ATTEMPT")
    logger.info("-" * 40)
    
    try:
        logger.info("Attempting to import open3d...")
        import open3d as o3d
        logger.success("✓ Open3D imported successfully!")
        logger.info(f"  Open3D version: {o3d.__version__}")
        logger.info(f"  Open3D location: {o3d.__file__}")
        
        # Try to create a simple point cloud
        try:
            pcd = o3d.geometry.PointCloud()
            logger.success("✓ Open3D point cloud creation successful")
        except Exception as pcd_error:
            logger.error(f"✗ Open3D point cloud creation failed: {pcd_error}")
            
    except ImportError as import_error:
        logger.error(f"✗ Open3D import failed with ImportError: {import_error}")
    except OSError as os_error:
        logger.error(f"✗ Open3D import failed with OSError: {os_error}")
        
        # Try to get more details about the missing library
        error_str = str(os_error)
        if 'libX11' in error_str:
            logger.error("  This is a libX11 library issue")
            
            # Try to find where libX11 should be
            try:
                result = subprocess.run(['find', '/usr', '-name', 'libX11.so*', '-type', 'f'], 
                                      capture_output=True, text=True, timeout=10)
                if result.stdout:
                    logger.info(f"  Found libX11 files at:")
                    for line in result.stdout.strip().split('\n'):
                        logger.info(f"    {line}")
                else:
                    logger.error("  No libX11 files found in /usr")
            except Exception as find_error:
                logger.error(f"  Could not search for libX11 files: {find_error}")
    except Exception as other_error:
        logger.error(f"✗ Open3D import failed with unexpected error: {other_error}")
        logger.error(f"  Error type: {type(other_error)}")
    
    # 8. System commands for additional info
    logger.info("\n8. SYSTEM COMMANDS OUTPUT")
    logger.info("-" * 40)
    
    commands = [
        ('ldd --version', 'Check dynamic linker version'),
        ('ldconfig -p | grep X11 | head -5', 'Check X11 in linker cache'),
        ('ldconfig -p | grep GL | head -5', 'Check GL in linker cache'),
        ('ldconfig -p | grep gdal | head -5', 'Check GDAL in linker cache'),
        ('which python3', 'Python3 location'),
        ('pip list | grep open3d', 'Open3D pip installation'),
        ('pip list | grep gdal', 'GDAL pip packages'),
    ]
    
    for cmd, description in commands:
        run_command(cmd, description)
    
    # Additional checks
    check_package_versions()
    check_gdal_comprehensive()
    check_library_dependencies()
    
    # Generate summary
    summary = generate_debug_summary()
    
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED CONTAINER ENVIRONMENT DEBUG COMPLETE")
    logger.info("=" * 80)
    
    return summary

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(sys.stdout, 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
               level="DEBUG")
    logger.add("enhanced_debug_container.log", 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
               level="DEBUG")
    
    summary = debug_container_environment()
    
    # Write summary to file
    with open("debug_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Debug complete!")