"""Sample GPU memory via nvidia-smi until a sentinel file appears.

We sample nvidia-smi rather than torch.cuda.max_memory_allocated() because the
latter only counts PyTorch's allocator. The real question for a 6 GB card is
total board usage: allocator + CUDA context + cuBLAS workspaces + the desktop.
"""
import subprocess, time, sys, os
from pathlib import Path

stop = Path(sys.argv[1]); out = Path(sys.argv[2]); interval = 1.0
rows, peak = [], 0
t0 = time.time()
while not stop.exists():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        used, total, util = [x.strip() for x in r.stdout.strip().split(",")]
        used = int(used); peak = max(peak, used)
        rows.append(f"{time.time()-t0:.1f},{used},{util}")
    except Exception as e:
        rows.append(f"{time.time()-t0:.1f},ERR,{e}")
    time.sleep(interval)
out.write_text("elapsed_s,mem_used_MiB,gpu_util_pct\n" + "\n".join(rows) +
               f"\n# PEAK_MiB={peak}\n# PEAK_GiB={peak/1024:.3f}\n")
print(f"PEAK GPU MEMORY: {peak} MiB = {peak/1024:.3f} GiB (board total 6141 MiB)")
