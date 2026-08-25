import time, torch, gsplat
print("gsplat", gsplat.__version__, flush=True)
t0 = time.time()
N = 1000
dev = "cuda"
means   = torch.randn(N,3,device=dev)
quats   = torch.randn(N,4,device=dev)
scales  = torch.rand(N,3,device=dev)*0.1
opac    = torch.rand(N,device=dev)
colors  = torch.rand(N,3,device=dev)
viewmat = torch.eye(4,device=dev)[None]
viewmat[0,2,3] = 8.0
K = torch.tensor([[[300.,0.,160.],[0.,300.,120.],[0.,0.,1.]]],device=dev)
print("--- triggering JIT compile of CUDA kernels (first run is slow) ---", flush=True)
img, alpha, meta = gsplat.rasterization(means,quats,scales,opac,colors,viewmat,K,320,240)
torch.cuda.synchronize()
print("COMPILE+RUN OK in %.1fs" % (time.time()-t0), flush=True)
print("image", tuple(img.shape), "dtype", img.dtype)
print("peak vram %.3f GiB" % (torch.cuda.max_memory_allocated()/2**30))
