"""Vectorised .ply -> .splat conversion (the upstream convert.py loops per-vertex).

.splat packs each Gaussian into 32 bytes:
    position  3 x float32  = 12
    scale     3 x float32  = 12
    color     4 x uint8    =  4   (RGB from the SH DC term + alpha)
    rotation  4 x uint8    =  4   (quaternion, quantised)
Note what this format THROWS AWAY: all 45 SH_rest coefficients, i.e. every
view-dependent appearance term. That is the honest cost of the 32-byte budget.
"""
import sys, time, numpy as np
from plyfile import PlyData

src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
ply = PlyData.read(src); v = ply["vertex"]
n = len(v); print(f"read {n:,} gaussians in {time.time()-t0:.1f}s")
print("ply fields:", len(v.data.dtype.names), "->", ", ".join(v.data.dtype.names[:12]), "...")

# sort by projected size * opacity, biggest first (viewer expects this ordering)
order = np.argsort(-np.exp(v["scale_0"]+v["scale_1"]+v["scale_2"]) / (1+np.exp(-v["opacity"])))

pos    = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)[order]
scale  = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32))[order]
rot    = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)[order]
rot   /= np.linalg.norm(rot, axis=1, keepdims=True)
SH_C0  = 0.28209479177387814
rgb    = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)[order]
alpha  = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))[order]

color  = np.clip(np.concatenate([rgb, alpha[:, None]], 1) * 255, 0, 255).astype(np.uint8)
rotq   = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

buf = np.zeros((n, 32), dtype=np.uint8)
buf[:,  0:12] = pos.view(np.uint8).reshape(n, 12)
buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
buf[:, 24:28] = color
buf[:, 28:32] = rotq
buf.tofile(dst)
print(f"wrote {dst} in {time.time()-t0:.1f}s total")
