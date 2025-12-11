"""
Detailed PyTorch + FKL JIT example.

This shows the complete workflow of:
1. Composing operations in Python
2. Generating kernel code dynamically
3. JIT compiling the kernel
4. Executing on PyTorch tensors
"""

import torch
import numpy as np
from fkl_ffi.pytorch import PyTorchFKL
from fkl_ffi.operations import Mul, Add, Crop, Resize

def show_kernel_generation():
    """Show how kernels are generated from operations."""
    print("=" * 70)
    print("FKL JIT Compilation Workflow")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fkl = PyTorchFKL(device=device)
    
    # Step 1: Compose operations
    print("\n1. User composes operations:")
    operations = [
        Mul(2.0),
        Add(1.0),
        Mul(0.5)
    ]
    for i, op in enumerate(operations, 1):
        print(f"   Operation {i}: {op.op_type}({op.params})")
    
    # Step 2: Generate kernel code
    print("\n2. FKL generates CUDA kernel code dynamically:")
    from fkl_ffi.operations import OperationChain
    chain = OperationChain(*operations)
    kernel_code = fkl._generate_kernel_code(chain, "example_kernel")
    print("   Generated kernel code:")
    print("   " + "-" * 60)
    for line in kernel_code.split('\n')[:15]:  # Show first 15 lines
        if line.strip():
            print(f"   {line}")
    print("   " + "-" * 60)
    print("   ... (full kernel code generated)")
    
    # Step 3: JIT compile
    print("\n3. JIT compiler compiles kernel at runtime:")
    print("   - Using NVRTC (NVIDIA Runtime Compilation)")
    print("   - Compiles CUDA code to PTX/CUBIN")
    print("   - No pre-compilation needed!")
    
    # Step 4: Execute
    print("\n4. Execute compiled kernel on GPU:")
    input_tensor = torch.randn(1000, device=device)
    kernel = fkl.compose_kernel(*operations)
    output = kernel(input_tensor)
    
    print(f"   ✓ Kernel '{kernel.kernel_name}' executed!")
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "=" * 70)
    print("Key Points:")
    print("=" * 70)
    print("• No static kernels - everything is generated at runtime")
    print("• Each operation combination = unique kernel")
    print("• Infinite combinations possible")
    print("• Automatic fusion and batching")
    print("• No CUDA knowledge required from user")


def demonstrate_kernel_caching():
    """Show kernel caching for performance."""
    print("\n" + "=" * 70)
    print("Kernel Caching")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fkl = PyTorchFKL(device=device)
    
    operations = [Mul(2.0), Add(1.0)]
    
    print("\nFirst call - kernel is compiled:")
    kernel1 = fkl.compose_kernel(*operations)
    print(f"   Kernel name: {kernel1.kernel_name}")
    print(f"   Cache size: {len(fkl._kernel_cache)}")
    
    print("\nSecond call - kernel is reused from cache:")
    kernel2 = fkl.compose_kernel(*operations)
    print(f"   Kernel name: {kernel2.kernel_name}")
    print(f"   Cache size: {len(fkl._kernel_cache)}")
    print(f"   Same kernel? {kernel1 is kernel2}")
    
    print("\nDifferent operations - new kernel compiled:")
    kernel3 = fkl.compose_kernel(Mul(3.0), Add(1.0))
    print(f"   Kernel name: {kernel3.kernel_name}")
    print(f"   Cache size: {len(fkl._kernel_cache)}")
    print(f"   Different kernel? {kernel3 is not kernel1}")


def show_automatic_fusion():
    """Show how operations automatically fuse."""
    print("\n" + "=" * 70)
    print("Automatic Fusion")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fkl = PyTorchFKL(device=device)
    
    print("\nTraditional approach (multiple kernels):")
    print("  Kernel 1: Read -> Mul(2.0) -> Write")
    print("  Kernel 2: Read -> Add(1.0) -> Write")
    print("  Kernel 3: Read -> Mul(0.5) -> Write")
    print("  = 3 kernel launches, 3 memory reads, 3 memory writes")
    
    print("\nFKL approach (single fused kernel):")
    print("  Single Kernel: Read -> Mul(2.0) -> Add(1.0) -> Mul(0.5) -> Write")
    print("  = 1 kernel launch, 1 memory read, 1 memory write")
    print("  = Data stays in registers between operations!")
    
    operations = [Mul(2.0), Add(1.0), Mul(0.5)]
    kernel = fkl.compose_kernel(*operations)
    
    print(f"\n✓ Fused kernel compiled: {kernel.kernel_name}")
    print("  All operations in one kernel - maximum efficiency!")


def main():
    """Run detailed examples."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. JIT compilation requires CUDA.")
        return
    
    try:
        show_kernel_generation()
        demonstrate_kernel_caching()
        show_automatic_fusion()
        
        print("\n" + "=" * 70)
        print("Summary: JIT Compilation Benefits")
        print("=" * 70)
        print("1. Dynamic kernel generation - no pre-compilation needed")
        print("2. Infinite combinations - each operation combo = new kernel")
        print("3. Automatic optimization - compiler optimizes fused code")
        print("4. Kernel caching - reuse compiled kernels for performance")
        print("5. No CUDA knowledge - users just compose operations")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

