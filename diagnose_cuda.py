"""
Diagnose CUDA availability and PyTorch setup.
"""

import torch
import sys

print("="*60)
print("PYTORCH CUDA DIAGNOSTICS")
print("="*60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"    - Capability: {props.major}.{props.minor}")
else:
    print("\nWARNING: CUDA not available!")
    print("\nPossible solutions:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. NVIDIA CUDA toolkit not installed on system")
    print("3. NVIDIA drivers not properly installed")
    print("\n" + "="*60)
    print("RECOMMENDED FIX:")
    print("="*60)
    print("\nUninstall current PyTorch and install CUDA-enabled version:")
    print("\n  pip uninstall torch torchvision torchaudio")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nOr for CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nThen verify with:")
    print("  python")
    print("  >>> import torch")
    print("  >>> torch.cuda.is_available()")
    print("  True  # Should return True now")

print("\n" + "="*60)
