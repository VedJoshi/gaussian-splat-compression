"""Download tandt_db.zip, extract ONLY tandt/truck, build images_2 + images_4.

gsplat's colmap Parser requires an `images_<factor>` folder to already exist
(examples/datasets/colmap.py:185 raises otherwise). Tanks&Temples ships only
`images/`, so we generate the downscaled folders ourselves. We write PNG so the
parser's jpg-rescale branch (colmap.py:193) is skipped and our images are used.
"""
import os, sys, time, zipfile, urllib.request
from pathlib import Path

URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
ROOT = Path(__file__).parent
ZIP = ROOT / "data" / "tandt_db.zip"
OUT = ROOT / "data"
ZIP.parent.mkdir(parents=True, exist_ok=True)

if not ZIP.exists():
    print(f"downloading {URL}")
    t0 = time.time()
    def hook(b, bs, total):
        done = b * bs
        pct = 100.0 * done / total if total else 0
        if b % 400 == 0:
            print(f"  {done/2**20:8.1f} MiB / {total/2**20:.1f} MiB  ({pct:5.1f}%)", flush=True)
    urllib.request.urlretrieve(URL, ZIP, reporthook=hook)
    print(f"downloaded in {time.time()-t0:.0f}s -> {ZIP.stat().st_size/2**30:.2f} GiB")
else:
    print(f"zip already present: {ZIP.stat().st_size/2**30:.2f} GiB")

print("extracting tandt/truck only ...")
with zipfile.ZipFile(ZIP) as z:
    members = [m for m in z.namelist() if m.startswith("tandt/truck/")]
    print(f"  {len(members)} members")
    z.extractall(OUT, members=members)

scene = OUT / "tandt" / "truck"
print("scene contents:", sorted(p.name for p in scene.iterdir()))

# build downscaled folders
from PIL import Image
src = scene / "images"
imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
print(f"  {len(imgs)} source images; first = {imgs[0].name} {Image.open(imgs[0]).size}")
for factor in (2, 4):
    dst = scene / f"images_{factor}"
    dst.mkdir(exist_ok=True)
    if len(list(dst.iterdir())) == len(imgs):
        print(f"  images_{factor} already built, skipping")
        continue
    t0 = time.time()
    for i, p in enumerate(imgs):
        im = Image.open(p)
        w, h = im.size
        im.resize((round(w / factor), round(h / factor)), Image.LANCZOS).save(dst / (p.stem + ".png"))
        if i % 60 == 0:
            print(f"    factor {factor}: {i}/{len(imgs)}", flush=True)
    print(f"  images_{factor} built in {time.time()-t0:.0f}s -> {Image.open(next(dst.iterdir())).size}")
print("DATA READY:", scene)
