"""Render the landing page's figures into site/figures/, which is committed.

    scripts\\env.bat, then: python -m scripts.site_figures   # needs CUDA and a built out/site
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.viewer_harness import REPO, launch, psnr, render_in_viewer, serve
from scripts.viewer_measure import site_target

FIGURES = REPO / "site" / "figures"
TRUCK = REPO / "out" / "truck"
# The cameras the landing-page cards open at; playroom's view 0 faces a bare wall.
CARD_VIEWS = {"truck": 0, "train": 0, "playroom": 12}
MAX_WIDTH = 1000
# Validated for the dark page surface (#1a1a19); baselines stay neutral grey.
GROUPS = {
    "Baselines": ("#8f8e88", "o", 40),
    "SH quantisation": ("#3987e5", "s", 40),
    "Pruning + SH quantisation": ("#d95926", "D", 40),
    "Deployed .splatc": ("#199e70", "*", 220),
}
LABELS = {
    "ply": ("PLY", "Baselines"),
    "splat": ("SPLAT", "Baselines"),
    "png": ("PngCompression", "Baselines"),
    "shvq256": ("256 entries", "SH quantisation"),
    "shvq1024": ("1,024 entries", "SH quantisation"),
    "shvq4096": ("4,096 entries", "SH quantisation"),
    "prune95-shvq4096": ("95% kept", "Pruning + SH quantisation"),
    "prune90-shvq4096": ("90% kept", "Pruning + SH quantisation"),
    "prune80-shvq4096": ("80% kept", "Pruning + SH quantisation"),
    "prune70-shvq4096": ("70% kept", "Pruning + SH quantisation"),
    "prune80-container": (".splatc (deployed)", "Deployed .splatc"),
}
DIRECT_LABELS = {"ply", "splat", "png", "prune80-container"}
# Lifted clear of the crowded 10-15 MB cluster.
LABEL_OFFSETS = {"prune80-container": (-4, 34)}
TEXT, MUTED, LINE = "#ececea", "#a8a7a0", "#3a3a38"


def truck_points() -> list[dict]:
    points: dict[str, dict] = {}
    for path in (TRUCK / "curve.json", TRUCK / "pruning" / "curve.json", TRUCK / "container-prune80" / "curve.json"):
        for point in json.loads(path.read_text(encoding="utf-8"))["points"]:
            # First file wins: the M3 run is the reference for the shared PLY, PNG and SH VQ points.
            if point["codec"] in LABELS and point["codec"] not in points:
                label, group = LABELS[point["codec"]]
                points[point["codec"]] = {"codec": point["codec"], "label": label, "group": group,
                                          "bytes": point["bytes"], "psnr": point["psnr"]}
    if missing := set(LABELS) - set(points):
        sys.exit(f"curve points missing: {sorted(missing)}")
    return list(points.values())


def plot_curve(points: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 9,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": LINE,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    figure, axes = plt.subplots(figsize=(7, 4.2))
    for group, (colour, marker, size) in GROUPS.items():
        members = sorted((p for p in points if p["group"] == group), key=lambda p: p["bytes"])
        if not members:
            continue
        x = [p["bytes"] / 1e6 for p in members]
        y = [p["psnr"] for p in members]
        if group != "Baselines" and len(members) > 1:
            axes.plot(x, y, color=colour, linewidth=2, zorder=2)
        axes.scatter(x, y, s=size, marker=marker, color=colour, edgecolors="#1a1a19", linewidths=1.5,
                     label=group, zorder=3)
        for p, px, py in zip(members, x, y):
            if p["codec"] in DIRECT_LABELS:
                offset = LABEL_OFFSETS.get(p["codec"])
                axes.annotate(p["label"], (px, py), textcoords="offset points", xytext=offset or (8, -3), color=TEXT,
                              arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8} if offset else None)
    axes.set_xscale("log")
    axes.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    axes.minorticks_off()
    axes.set_xlabel("File size (MB, log scale)")
    axes.set_ylabel("Held-out PSNR (dB)")
    axes.grid(True, which="major", color=LINE, linewidth=0.6)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.legend(frameon=False, loc="lower right", labelcolor=TEXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(figure)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def stills() -> dict:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from splatpipe.bench.cameras import load_val_views
    from splatpipe.bench.render import render_view
    from splatpipe.gaussians import read_ply
    from splatpipe.paths import RunPaths

    record = {}
    with sync_playwright() as playwright:
        browser = launch(playwright)
        for scene, index in CARD_VIEWS.items():
            target = site_target(scene)
            view = load_val_views(target.data, data_factor=1, test_every=8)[index]
            height, width = view.image.shape[:2]
            reference = view.image.astype(np.float32) / 255.0
            original = _to_uint8(render_view(read_ply(RunPaths.for_run(REPO / "out", scene).ply), view))
            with serve(target.root) as base:
                compressed, errors = render_in_viewer(browser, base, target.url, view.camtoworld, view.K, width, height)
            if errors:
                sys.exit(f"viewer console errors on {scene}: {errors}")
            for kind, image in (("photo", view.image), ("original", original), ("compressed", compressed)):
                picture = Image.fromarray(image)
                if picture.width > MAX_WIDTH:
                    picture = picture.resize((MAX_WIDTH, round(picture.height * MAX_WIDTH / picture.width)), Image.LANCZOS)
                picture.save(FIGURES / f"{scene}-{kind}.jpg", quality=85)
            record[scene] = {
                "view": index, "width": picture.width, "height": picture.height,
                "original_db": psnr(original, reference), "compressed_db": psnr(compressed, reference),
            }
        browser.close()
    return record


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_curve(truck_points(), FIGURES / "curve.svg")
    record = stills()
    (FIGURES / "stills.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
