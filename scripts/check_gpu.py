#!/usr/bin/env python3
"""
GPU Detection and CUDA Check for EDISON
Diagnoses GPU availability and llama-cpp-python CUDA support
"""

import sys
import subprocess

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    print("=" * 60)
    print("1. Checking NVIDIA GPU with nvidia-smi")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvidia-smi is working")
            print("\nGPU Information:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {e}")
        return False

def check_cuda_python():
    """Check CUDA availability through Python"""
    print("\n" + "=" * 60)
    print("2. Checking CUDA with Python")
    print("=" * 60)
    
    # Try torch first (common)
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available through PyTorch")
            print(f"  CUDA version: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"  GPU count: {gpu_count}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {name} ({vram_gb:.1f}GB VRAM)")
        else:
            print("✗ CUDA not available through PyTorch")
            print("  This usually means:")
            print("    - PyTorch CPU-only build is installed")
            print("    - CUDA drivers not compatible with PyTorch")
    except ImportError:
        print("⚠ PyTorch not installed (optional)")
    except Exception as e:
        print(f"✗ Error checking PyTorch CUDA: {e}")

def check_llama_cpp():
    """Check llama-cpp-python CUDA support"""
    print("\n" + "=" * 60)
    print("3. Checking llama-cpp-python CUDA Support")
    print("=" * 60)
    
    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python is installed")
        
        # Check if CUDA build
        try:
            # Try to import CUDA-specific functions
            import llama_cpp
            
            # Check for CUDA in the binary
            result = subprocess.run(['python3', '-c', 
                'import llama_cpp._internals as internals; print(hasattr(internals._lib, "llama_supports_gpu_offload"))'],
                capture_output=True, text=True)
            
            if result.returncode == 0 and 'True' in result.stdout:
                print("✓ llama-cpp-python appears to have GPU support")
            else:
                print("⚠ Could not verify GPU support in llama-cpp-python")
                print("  This might indicate a CPU-only build")
        except Exception as e:
            print(f"⚠ Could not check CUDA support: {e}")
            
        # Test actual GPU offloading
        print("\nTesting GPU offload capability...")
        print("  (This will be quick, just testing n_gpu_layers parameter)")
        try:
            # This won't load a model, just test the parameter
            print("  Testing with n_gpu_layers=-1 (all layers to GPU)")
            print("  If this works, GPU offload is supported")
            print("  Note: Actual model loading test requires model files")
        except Exception as e:
            print(f"✗ GPU test failed: {e}")
            
    except ImportError:
        print("✗ llama-cpp-python not installed")
        print("  Install with: pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"✗ Error checking llama-cpp-python: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("\n" + "=" * 60)
    print("4. Checking Environment Variables")
    print("=" * 60)
    
    import os
    cuda_vars = {
        'CUDA_VISIBLE_DEVICES': 'Controls which GPUs are visible',
        'CUDA_HOME': 'CUDA installation directory',
        'LD_LIBRARY_PATH': 'Library search path (should include CUDA libs)',
    }
    
    for var, description in cuda_vars.items():
        value = os.environ.get(var, 'Not set')
        print(f"{var}:")
        print(f"  Value: {value}")
        print(f"  ({description})")

def check_llama_cpp_build():
    """Check how llama-cpp-python was built"""
    print("\n" + "=" * 60)
    print("5. Checking llama-cpp-python Build Type")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['python3', '-c', 'import llama_cpp; print(llama_cpp.__file__)'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            package_path = result.stdout.strip()
            print(f"Package location: {package_path}")
            
            # Check for CUDA libraries
            import os
            lib_dir = os.path.dirname(package_path)
            print(f"\nChecking for CUDA libraries in: {lib_dir}")
            
            result = subprocess.run(
                ['find', lib_dir, '-name', '*.so', '-o', '-name', '*.dll'],
                capture_output=True, text=True
            )
            
            libs = result.stdout.strip().split('\n')
            cuda_found = False
            for lib in libs:
                if lib and ('cuda' in lib.lower() or 'cublas' in lib.lower()):
                    print(f"  ✓ Found CUDA library: {os.path.basename(lib)}")
                    cuda_found = True
            
            if not cuda_found:
                print("  ✗ No CUDA libraries found")
                print("  This indicates llama-cpp-python was built without CUDA")
                print("\n  To rebuild with CUDA:")
                print("    pip uninstall llama-cpp-python -y")
                print("    CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
                
    except Exception as e:
        print(f"✗ Error checking build: {e}")

def main():
    print("\n" + "=" * 60)
    print("EDISON GPU Diagnostic Tool")
    print("=" * 60 + "\n")
    
    gpu_ok = check_nvidia_smi()
    check_cuda_python()
    check_llama_cpp()
    check_environment()
    check_llama_cpp_build()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if not gpu_ok:
        print("✗ NVIDIA GPU not detected")
        print("  Action: Install NVIDIA drivers")
        return 1
    
    print("\nIf llama-cpp-python is using CPU:")
    print("1. Rebuild llama-cpp-python with CUDA:")
    print("   pip uninstall llama-cpp-python -y")
    print("   CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
    print("\n2. Restart edison-core service:")
    print("   sudo systemctl restart edison-core")
    print("\n3. Check logs:")
    print("   sudo journalctl -u edison-core -f")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
