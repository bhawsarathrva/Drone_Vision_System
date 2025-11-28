"""
PyTorch DLL Error Fix Script
Diagnoses and provides solutions for PyTorch DLL initialization errors on Windows
"""
import sys
import os
import subprocess
from pathlib import Path

def check_pytorch_installation():
    """Check PyTorch installation"""
    print("="*70)
    print("PyTorch Installation Diagnostics")
    print("="*70)
    
    try:
        import torch
        print(f"✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
        return True
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def check_dll_dependencies():
    """Check for required DLL files"""
    print("\n" + "="*70)
    print("Checking DLL Dependencies")
    print("="*70)
    
    try:
        import torch
        torch_path = Path(torch.__file__).parent
        dll_path = torch_path / "lib"
        
        required_dlls = ['c10.dll', 'torch_cpu.dll', 'torch_python.dll']
        
        for dll in required_dlls:
            dll_file = dll_path / dll
            if dll_file.exists():
                print(f"✓ Found: {dll}")
            else:
                print(f"✗ Missing: {dll}")
        
        return True
    except Exception as e:
        print(f"✗ Could not check DLLs: {e}")
        return False

def get_python_info():
    """Get Python environment info"""
    print("\n" + "="*70)
    print("Python Environment")
    print("="*70)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Virtual Environment: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}")

def print_solutions():
    """Print potential solutions"""
    print("\n" + "="*70)
    print("SOLUTIONS TO FIX PYTORCH DLL ERROR")
    print("="*70)
    
    print("\n🔧 Solution 1: Reinstall PyTorch (RECOMMENDED)")
    print("-" * 70)
    print("Uninstall and reinstall PyTorch with the correct version:")
    print()
    print("# Uninstall current PyTorch")
    print("pip uninstall torch torchvision torchaudio -y")
    print()
    print("# Install PyTorch CPU version (stable)")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("# OR install with UV (faster)")
    print("uv pip uninstall torch torchvision torchaudio -y")
    print("uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n🔧 Solution 2: Install Visual C++ Redistributables")
    print("-" * 70)
    print("PyTorch requires Visual C++ Redistributables:")
    print()
    print("Download and install from:")
    print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("Or run this PowerShell command:")
    print('Start-Process "https://aka.ms/vs/17/release/vc_redist.x64.exe"')
    
    print("\n🔧 Solution 3: Use Conda (Alternative)")
    print("-" * 70)
    print("If you have Anaconda/Miniconda:")
    print()
    print("conda install pytorch torchvision torchaudio cpuonly -c pytorch")
    
    print("\n🔧 Solution 4: Create Fresh Virtual Environment")
    print("-" * 70)
    print("# Deactivate current venv")
    print("deactivate")
    print()
    print("# Remove old venv")
    print('Remove-Item -Recurse -Force venv')
    print()
    print("# Create new venv")
    print("python -m venv venv_new")
    print()
    print("# Activate")
    print(".\\venv_new\\Scripts\\Activate.ps1")
    print()
    print("# Install PyTorch")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("# Install other requirements")
    print("pip install ultralytics opencv-python loguru")
    
    print("\n🔧 Solution 5: Quick Fix Script")
    print("-" * 70)
    print("Run the automated fix:")
    print()
    print("python fix_pytorch.py --auto")
    
    print("\n" + "="*70)
    print("RECOMMENDED: Try Solution 1 first (Reinstall PyTorch)")
    print("="*70)

def auto_fix():
    """Attempt automatic fix"""
    print("\n" + "="*70)
    print("Attempting Automatic Fix")
    print("="*70)
    
    print("\nStep 1: Uninstalling PyTorch...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      check=True)
        print("✓ PyTorch uninstalled")
    except Exception as e:
        print(f"✗ Uninstall failed: {e}")
        return False
    
    print("\nStep 2: Installing PyTorch CPU version...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                       "--index-url", "https://download.pytorch.org/whl/cpu"], 
                      check=True)
        print("✓ PyTorch installed")
    except Exception as e:
        print(f"✗ Install failed: {e}")
        return False
    
    print("\nStep 3: Verifying installation...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} working!")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix PyTorch DLL Error')
    parser.add_argument('--auto', action='store_true', help='Attempt automatic fix')
    parser.add_argument('--check', action='store_true', help='Only check installation')
    
    args = parser.parse_args()
    
    get_python_info()
    
    if args.check:
        check_pytorch_installation()
        check_dll_dependencies()
        return
    
    if args.auto:
        success = auto_fix()
        if success:
            print("\n✓ Fix completed successfully!")
        else:
            print("\n✗ Automatic fix failed. Please try manual solutions.")
            print_solutions()
    else:
        # Run diagnostics
        pytorch_ok = check_pytorch_installation()
        check_dll_dependencies()
        
        if not pytorch_ok:
            print_solutions()

if __name__ == "__main__":
    main()
