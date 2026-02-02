"""
Python Sandbox - Isolated code execution using Docker
Enables ChatGPT-style code interpreter for data analysis, plots, calculations
"""

import docker
import uuid
import tempfile
import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import re

logger = logging.getLogger(__name__)


class PythonSandbox:
    """Isolated Python code execution sandbox using Docker"""
    
    def __init__(self, 
                 image: str = "python:3.11-slim",
                 memory_limit: str = "512m",
                 cpu_quota: int = 100000,  # 1 CPU core
                 timeout: int = 30):
        """
        Initialize Python sandbox
        
        Args:
            image: Docker image to use
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_quota: CPU quota (100000 = 1 core)
            timeout: Execution timeout in seconds
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.timeout = timeout
        
        try:
            self.client = docker.from_env()
            # Ensure Python image is available
            self._ensure_image()
            logger.info(f"✓ Python sandbox initialized with image {image}")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker not available: {e}")
    
    def _ensure_image(self):
        """Ensure the Python image is pulled"""
        try:
            self.client.images.get(self.image)
            logger.debug(f"Image {self.image} already available")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)
            logger.info(f"✓ Image {self.image} pulled successfully")
    
    def _install_packages(self, packages: List[str]) -> str:
        """
        Generate pip install command for packages
        
        Args:
            packages: List of package names to install
            
        Returns:
            Shell command to install packages
        """
        if not packages:
            return ""
        
        # Common data science packages
        safe_packages = [
            "numpy", "pandas", "matplotlib", "scipy", "scikit-learn",
            "seaborn", "plotly", "pillow", "requests", "beautifulsoup4"
        ]
        
        # Filter to only safe packages
        packages_to_install = [p for p in packages if p.lower() in safe_packages]
        
        if not packages_to_install:
            return ""
        
        return f"pip install -q {' '.join(packages_to_install)} && "
    
    async def execute_code(self, 
                          code: str, 
                          packages: Optional[List[str]] = None,
                          files: Optional[Dict[str, bytes]] = None) -> Dict[str, Any]:
        """
        Execute Python code in isolated Docker container
        
        Args:
            code: Python code to execute
            packages: Optional list of packages to install (numpy, pandas, etc.)
            files: Optional dict of filename -> file content (for data files)
            
        Returns:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "images": List[str],  # Base64-encoded PNG images
                "execution_time": float,
                "error": Optional[str]
            }
        """
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"Executing code in sandbox {execution_id}")
        
        # Create temporary directory for code and outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write code to file
            code_file = tmpdir_path / "script.py"
            code_file.write_text(code)
            
            # Write input files if provided
            if files:
                for filename, content in files.items():
                    (tmpdir_path / filename).write_bytes(content)
            
            # Create output directory for plots
            output_dir = tmpdir_path / "output"
            output_dir.mkdir()
            
            # Wrap code to save matplotlib plots
            wrapped_code = f"""
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Redirect plot saving
_original_show = plt.show
_plot_counter = [0]

def _save_plot(*args, **kwargs):
    _plot_counter[0] += 1
    filename = f'/workspace/output/plot_{{_plot_counter[0]}}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close('all')

plt.show = _save_plot

# Execute user code
try:
    exec(open('/workspace/script.py').read())
except Exception as e:
    import traceback
    print(f"ERROR: {{e}}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
"""
            
            wrapper_file = tmpdir_path / "wrapper.py"
            wrapper_file.write_text(wrapped_code)
            
            # Build command
            install_cmd = self._install_packages(packages or [])
            cmd = f"{install_cmd}python /workspace/wrapper.py"
            
            try:
                # Run container
                container = self.client.containers.run(
                    self.image,
                    command=["sh", "-c", cmd],
                    volumes={str(tmpdir_path): {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir="/workspace",
                    mem_limit=self.memory_limit,
                    cpu_quota=self.cpu_quota,
                    network_mode="none",  # No network access
                    remove=True,
                    detach=True
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=self.timeout)
                    logs = container.logs()
                    stdout = logs.decode('utf-8', errors='replace')
                    stderr = ""
                    
                    # Check exit code
                    exit_code = result.get('StatusCode', -1)
                    success = exit_code == 0
                    
                    if not success:
                        stderr = stdout  # Errors go to stdout in this setup
                        stdout = ""
                    
                except Exception as e:
                    # Timeout or other error
                    try:
                        container.kill()
                    except:
                        pass
                    
                    execution_time = time.time() - start_time
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Execution timeout after {self.timeout}s",
                        "images": [],
                        "execution_time": execution_time,
                        "error": f"Timeout: {str(e)}"
                    }
                
                # Collect generated images
                images = []
                for img_file in output_dir.glob("*.png"):
                    try:
                        img_data = img_file.read_bytes()
                        img_b64 = base64.b64encode(img_data).decode('utf-8')
                        images.append(img_b64)
                    except Exception as e:
                        logger.warning(f"Failed to read image {img_file}: {e}")
                
                execution_time = time.time() - start_time
                
                result = {
                    "success": success,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                    "images": images,
                    "execution_time": execution_time,
                    "error": None if success else stderr.strip()
                }
                
                logger.info(f"Sandbox {execution_id} completed in {execution_time:.2f}s, "
                          f"{len(images)} images generated")
                
                return result
                
            except docker.errors.ContainerError as e:
                execution_time = time.time() - start_time
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "images": [],
                    "execution_time": execution_time,
                    "error": f"Container error: {str(e)}"
                }
            
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Sandbox execution error: {e}")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "images": [],
                    "execution_time": execution_time,
                    "error": f"Execution failed: {str(e)}"
                }
    
    def health_check(self) -> bool:
        """Check if Docker is available and working"""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Docker health check failed: {e}")
            return False


# Global sandbox instance
_sandbox_instance: Optional[PythonSandbox] = None


def get_sandbox() -> PythonSandbox:
    """Get or create global sandbox instance"""
    global _sandbox_instance
    if _sandbox_instance is None:
        _sandbox_instance = PythonSandbox()
    return _sandbox_instance


async def execute_python_code(code: str, 
                               packages: Optional[List[str]] = None,
                               files: Optional[Dict[str, bytes]] = None) -> Dict[str, Any]:
    """
    Convenience function to execute Python code
    
    Args:
        code: Python code to execute
        packages: Optional list of packages to install
        files: Optional input files
        
    Returns:
        Execution result dictionary
    """
    sandbox = get_sandbox()
    return await sandbox.execute_code(code, packages, files)
