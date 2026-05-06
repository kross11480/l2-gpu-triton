# Triton GEMM & 1×1 Convolution Lab

A GPU programming lab implemented as a single Jupyter notebook (`l2-triton_matmul.ipynb`) that builds an optimized matrix-multiplication kernel in [Triton](https://github.com/triton-lang/triton), reuses it for 1×1 convolution, and benchmarks both against cuBLAS/cuDNN.

**Requires a CUDA GPU.** The notebook cannot run on macOS or CPU-only machines.

## Aims

1. Write an optimized matrix multiplication on GPU using Triton.
2. Reuse that kernel to implement a **1×1 convolution** — the dominant op in depthwise-separable CNNs (MobileNet-v1, EfficientNet, etc.).
3. Benchmark throughput against `torch.matmul` (cuBLAS) and `torch.nn.functional.conv2d` (cuDNN).

## Setup

### Platforms

| Platform | GPU | How to enable |
|---|---|---|
| Google Colab | free T4 | Runtime → Change runtime type → GPU |
| Kaggle Notebooks | free T4 / P100 | Settings → Accelerator → GPU |
| Local CUDA box | any NVIDIA | see below |

### Local installation (CUDA 12.4)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install triton torchprofile -q
conda install -c nvidia nsight-compute -y
export PATH=/opt/conda/nsight-compute/2024.1.1/target/linux-desktop-glibc_2_11_3-x64/:$PATH
```

The first notebook cell installs dependencies and confirms the device:

```python
!pip install triton torchprofile -q
DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(DEVICE)  # should print: cuda
```

## Architecture

### Kernels (`@triton.jit`)

| Kernel | Grid shape | What it computes |
|---|---|---|
| `simple_matmul_kernel` | `(N, N)` — one thread per output element | Naive scalar matmul, no tiling |
| `simple_tiled_matmul_kernel` | `(N//BLOCK)²` flat 1-D grid | Tiled matmul; `BLOCK` is a `tl.constexpr` |

Key Triton patterns used:
- `tl.program_id(axis)` — block-index computation
- `tl.arange(0, BLOCK)` + broadcasting — tile pointer arithmetic
- `tl.dot(a, b)` — tile-level GEMM (maps to tensor cores when available)
- `tl.zeros((BLOCK, BLOCK), dtype=tl.float32)` — accumulator

### Benchmarking utilities

- `bench(fn, warmup, reps)` — mean GPU wall-clock time in ms
- `gflops(N, ms)` / `gflops_gemm(M, N, K, ms)` — compute throughput
- `kernel_info(jit_fn)` — extracts register count, shared memory, warp count, and pipeline stages from the Triton compiled-kernel cache
- `triton.testing.perf_report` / `do_bench` — official Triton benchmark harness for the final TFLOPS comparison plot

### Unit testing

Correctness is verified with `torch.testing.assert_close` against `torch.matmul`. Tolerances are loose (`atol=1e-1, rtol=1e-2`) because FP32 tile accumulation introduces numerical drift.

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `BLOCK` | `64` | Tile size (power of 2). Values beyond ~128 may exceed shared-memory capacity. |
| `num_warps` | `4` | Best occupancy on T4/V100 for `BLOCK=64`. |
| Matrix sizes | `256*i, i∈[2,15]` | Sweep used in benchmarks. |

## Notes

- Output tensors accumulate in FP32 internally; final results can be cast to FP16 as needed.
- Nsight profiling (`ncu`) may be locked on shared GPU clusters — use a personal device (e.g. Jetson, local workstation) for detailed profiling.