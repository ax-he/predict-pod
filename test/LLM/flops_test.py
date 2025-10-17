import torch, time
assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = True  # 如需 FP32 严格禁止 TF32 则设 False
device, dtype = "cuda", torch.float32        # 可改成 torch.float16 / bfloat16
M = N = K = 8192

A = torch.randn((M, K), device=device, dtype=dtype)
B = torch.randn((K, N), device=device, dtype=dtype)

# 预热
for _ in range(5):
    _ = A @ B
torch.cuda.synchronize()

best = 1e9
for _ in range(5):
    t0 = time.perf_counter()
    C = A @ B
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    best = min(best, dt)

flops = 2.0 * M * N * K / best
print(f"Size={M}x{K}x{N}, best={best:.4f}s, ~{flops/1e12:.2f} TFLOP/s")
