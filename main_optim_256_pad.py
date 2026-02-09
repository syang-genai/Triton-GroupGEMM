from typing import Optional
import time
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"
def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count 

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'NUM_SM':num_sms() ,
            'num_warps': 4,
            'num_stages': 4,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),

        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),

        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
            'num_warps': 4,
            'num_stages': 4,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):  
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3 + 0)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            
            lda = tl.load(g_lds + g * 3 + 0)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            
            full_m = (tile_m_idx + 1) * BLOCK_SIZE_M <= gm
            full_n = (tile_n_idx + 1) * BLOCK_SIZE_N <= gn
            full_tile = full_m & full_n


            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            if not full_tile:
                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
                
                a_mask = (offs_am < gm)[:, None] & (offs_k < gk)[None, :]
                b_mask = (offs_k < gk)[:, None] & (offs_bn < gn)[None, :]


                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    tl.multiple_of(a_ptrs, [16, 16])
                    tl.multiple_of(b_ptrs, [16, 16])
                    a = tl.load(a_ptrs, mask=a_mask)
                    b = tl.load(b_ptrs, mask=b_mask)
                    
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K * ldb
                
                c = accumulator.to(tl.float16)
                offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                
                c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
                c_mask = (offs_cm < gm)[:, None] & (offs_cn < gn)[None, :]
                
                tl.store(c_ptrs, c, mask=c_mask)
            else:
                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
                b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    tl.multiple_of(a_ptrs, [16, 16])
                    tl.multiple_of(b_ptrs, [16, 16])
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K * ldb
                
                c = accumulator.to(tl.float16)
                offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
    
                tl.store(c_ptrs, c)
            
            tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles
    return


def group_gemm_fn(group_A, group_B, pad_multiple=256):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C_pad = []  
    original_M = [] 
    padded_group_A = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape

        if M < pad_multiple:
            M_pad = pad_multiple
            A_pad = torch.zeros((M_pad, K), device=A.device, dtype=A.dtype)
            A_pad[:M, :] = A
            C_pad = torch.empty((M_pad, N), device=A.device, dtype=A.dtype)
        else:
            M_pad = M
            A_pad = A
            C_pad = torch.empty((M_pad, N), device=A.device, dtype=A.dtype)

        padded_group_A.append(A_pad)
        group_C_pad.append(C_pad)
        original_M.append(M)

        A_addrs.append(A_pad.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C_pad.data_ptr())
        g_sizes += [M_pad, N, K]
        g_lds += [A_pad.stride(0), B.stride(0), C_pad.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )

    group_C = [
        C_pad[:M, :]
        for C_pad, M in zip(group_C_pad, original_M)
    ]
    return group_C


def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
    )

def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[i for i in range(251,260)], 
        line_arg='provider',
        line_vals=['cublas', 'triton'],
        line_names=["cuBLAS", "Triton"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="GB/s",  
        plot_name="group-gemm-performance-optim-pow2-pad-gbps",
        args={},
    ))
def benchmark_batches_gbps(M, provider):
    N = 4096
    K = 2048
    group_size = 8
    pad_multiple = 256  
    group_A = []
    group_B = []
    group_C_pad = [] 
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    original_M = [] 

    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)

        if M < pad_multiple:
            M_pad = pad_multiple
            A_pad = torch.zeros((M_pad, K), device=DEVICE, dtype=A.dtype)
            A_pad[:M, :] = A
            C_pad = torch.empty((M_pad, N), device=DEVICE, dtype=A.dtype)
        else:
            M_pad = M
            A_pad = A
            C_pad = torch.empty((M_pad, N), device=DEVICE, dtype=A.dtype)

        group_A.append(A_pad)
        group_B.append(B)
        group_C_pad.append(C_pad)
        original_M.append(M)

        A_addrs.append(A_pad.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C_pad.data_ptr())
        g_sizes += [M_pad, N, K]
        g_lds += [A_pad.stride(0), B.stride(0), C_pad.stride(0)]


    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.25, 0.75]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)


    # --- calculate FLOPS ---
    total_flops = sum([2 * m * n * k for m,n, k in zip(original_M, [N]*group_size, [K]*group_size)])
    
    # --- calculate bandwidth (bytes accessed) ---
    total_bytes = sum([
        (A.numel() + B.numel() + C_pad.numel()) * 2
        for A, B, C_pad in zip(group_A, group_B, group_C_pad)
    ])

    perf_gflops = lambda t: total_flops / (t * 1e6)
    perf_gbps = lambda t: total_bytes / (t * 1e6)
    return perf_gbps(ms), perf_gbps(max_ms), perf_gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[i for i in range(251,260)], 
        line_arg='provider',
        line_vals=['cublas', 'triton'],
        line_names=["cuBLAS", "Triton"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="GFlops",  
        plot_name="group-gemm-performance-optim-pow2-pad-gflops",
        args={},
    ))
def benchmark_batches_gflops(M, provider):
    N = 4096
    K = 2048
    group_size = 8
    pad_multiple = 256  
    group_A = []
    group_B = []
    group_C_pad = [] 
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    original_M = [] 

    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)

        if M < pad_multiple:
            M_pad = pad_multiple
            A_pad = torch.zeros((M_pad, K), device=DEVICE, dtype=A.dtype)
            A_pad[:M, :] = A
            C_pad = torch.empty((M_pad, N), device=DEVICE, dtype=A.dtype)
        else:
            M_pad = M
            A_pad = A
            C_pad = torch.empty((M_pad, N), device=DEVICE, dtype=A.dtype)

        group_A.append(A_pad)
        group_B.append(B)
        group_C_pad.append(C_pad)
        original_M.append(M)

        A_addrs.append(A_pad.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C_pad.data_ptr())
        g_sizes += [M_pad, N, K]
        g_lds += [A_pad.stride(0), B.stride(0), C_pad.stride(0)]


    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.25, 0.75]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)

    # --- calculate FLOPS ---
    total_flops = sum([2 * m * n * k for m,n, k in zip(original_M, [N]*group_size, [K]*group_size)])
    
    # --- calculate bandwidth (bytes accessed) ---
    # FP16 = 2 bytes
    total_bytes = sum([
        (A.numel() + B.numel() + C_pad.numel()) * 2
        for A, B, C_pad in zip(group_A, group_B, group_C_pad)
    ])
    perf_gflops = lambda t: total_flops / (t * 1e6)
    perf_gbps = lambda t: total_bytes / (t * 1e6)
    return perf_gflops(ms), perf_gflops(max_ms), perf_gflops(min_ms)


def main():
    group_m = [i for i in range(251,260)]
    group_n = 4096
    group_k = 2048
    group_A = []
    group_B = []
    
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n
        K = group_k
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
    
    tri_out = group_gemm_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(group_size):
        assert torch.allclose(ref_out[i], tri_out[i], atol=1e-3, rtol=1e-3), "numeric error"
        

    benchmark_batches_gbps.run(show_plots=True, print_data=True, save_path="group-gemm-performance-optim-256-pad")
    benchmark_batches_gflops.run(show_plots=True, print_data=True, save_path="group-gemm-performance-optim-256-pad")

if __name__=="__main__":
    main()