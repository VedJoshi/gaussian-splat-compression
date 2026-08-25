# Getting up to speed

Written for where you actually are: comfortable with PyTorch, transformers and training loops from
the FYP, but `ml-foundations/06_3d_cv/` is an empty folder. So this skips machine learning basics
and spends its time on the geometry and rendering you have not done yet.

Ordered so that each part is useful on its own. If you stop after part 2 you will still understand
what the spike did.

A note on links: arXiv IDs are given where I am confident of them. Where I only give a title and
author, search for it rather than trusting me to have the number right.

## Part 0: the one-paragraph version

A Gaussian splat scene is a few million fuzzy 3D blobs. Each one has a position, a shape (scale plus
rotation), a transparency, and a colour that changes depending on the angle you view it from. To
render, you project every blob onto the image plane and blend them back to front. Nothing is a
neural network. "Training" means gradient descent directly on the blobs' numbers until the rendered
images match the photos you took. That is why it is fast, and it is why the output file is enormous:
you are literally storing millions of blobs.

## Part 1: camera geometry (the real gap)

You cannot follow any of this without knowing what a camera matrix is. This is the prerequisite
everything else assumes.

What to learn: pinhole camera model, intrinsics `K`, extrinsics (world to camera), homogeneous
coordinates, projection, lens distortion, and roughly what Structure from Motion does.

- Cyrill Stachniss, *Photogrammetry* lecture series on YouTube. The best free source for this
  material. Watch the camera model and orientation lectures. Slow and thorough.
- Shree Nayar, *First Principles of Computer Vision*, YouTube (Columbia). Short, sharp episodes.
  The camera and stereo playlists are what you want.
- Szeliski, *Computer Vision: Algorithms and Applications*, free at szeliski.org/Book. Chapter 2 for
  image formation, chapter 11 for structure from motion. Use as reference, not cover to cover.
- Hartley and Zisserman, *Multiple View Geometry*. The standard reference. Dense. Look things up in
  it, do not read it front to back.
- Schönberger and Frahm, *Structure-from-Motion Revisited*, CVPR 2016. This is the COLMAP paper. You
  ran COLMAP output in the spike without knowing what produced it; this is what produced it.

Rough time: 8 to 12 hours to be functional. This is the part that pays off across all of 3D vision,
not just splatting.

Concrete check that you have it: explain why the spike's COLMAP file says 1957x1091 while the images
on disk are 979x546, and why dividing `K` by 2 fixes it. If that sentence makes sense, you are done
with part 1.

## Part 2: radiance fields, then splatting

Do these in order. NeRF first, even though it is superseded, because 3DGS is written as a reaction
to it and the vocabulary comes from there.

- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*,
  ECCV 2020, arXiv:2003.08934. Read for the idea of volume rendering and novel view synthesis. You
  do not need the positional encoding details.
- Barron et al., *Mip-NeRF 360*, arXiv:2111.12077. Skim. Mainly so you recognise the dataset name,
  since it is the standard benchmark you will be compared against.
- Kerbl, Kopanas, Leimkühler and Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering*, SIGGRAPH 2023, arXiv:2308.04079. **Read this one properly.** It is the paper the
  entire spike is built on. Project page at repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ has
  the video, which is worth watching first.

When you read the 3DGS paper, the four things to actually understand are:

1. How a 3D Gaussian projects to a 2D one (the covariance is transformed, not the shape directly).
2. Why the covariance is stored as separate scale and rotation instead of a raw matrix. The reason
   is that gradient descent would otherwise drift into non positive semi-definite matrices.
3. Adaptive density control: how splats get cloned, split and pruned during training.
4. Tile-based rasterisation and back to front alpha blending, and why sorting is the expensive part.

- Kheradmand et al., *3D Gaussian Splatting as Markov Chain Monte Carlo*, NeurIPS 2024,
  arXiv:2404.09591. This is the `mcmc` strategy the spike actually used, and the reason
  `--strategy.cap-max` exists. Read after the main paper.

Rough time: 10 to 15 hours including the video and a re-read.

## Part 3: spherical harmonics

Worth its own section because it is 76 percent of the file size and the main target if you do
compression work. It is also the piece people skip and then cannot reason about.

What to learn: SH as a basis for functions on a sphere, what "degree" means, why degree 3 gives 16
coefficients per colour channel and therefore 48 floats, and what the DC term alone represents
(a flat, view-independent colour).

- Peter-Pike Sloan, *Stupid Spherical Harmonics Tricks*. The classic practical write-up, aimed at
  graphics programmers rather than mathematicians.
- Any real-time rendering course section on irradiance environment maps covers the same ground.
- For intuition, Fourier series on a circle generalised to a sphere. If you are comfortable with
  Fourier you are most of the way there already.

Concrete check: explain why `.splat` files are 32 bytes per Gaussian while `.ply` files are 236, and
what visually disappears as a result.

Rough time: 3 to 4 hours.

## Part 4: compression, if the project goes that way

This is the current research frontier and where the portfolio work would sit. Read after parts 2
and 3, not before.

- Aras Pranckevičius wrote a blog series on making Gaussian splats smaller, starting at
  aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/. Start here rather than with a paper.
  It is an experienced graphics engineer working through the problem from scratch, with numbers at
  every step, and it is much closer to what you would actually build than an academic paper is.
- Niedermayr, Stumpfegger and Westermann, *Compressed 3D Gaussian Splatting for Accelerated Novel
  View Synthesis*, CVPR 2024, arXiv:2401.02436. Vector quantisation and entropy coding.
- Lee et al., *Compact 3D Gaussian Representation for Radiance Field*, CVPR 2024, arXiv:2311.13681.
  Learned masking plus compact appearance representation.
- Fan et al., *LightGaussian*, arXiv:2311.17245. Pruning by contribution, plus SH distillation.
- Lu et al., *Scaffold-GS*, arXiv:2312.00109. A structurally different approach: anchor points that
  generate Gaussians, rather than storing every Gaussian.
- gsplat's own `gsplat/compression/` source. Read the code for `PngCompression`. It is the baseline
  you would be measured against, and reading it is faster than reading a paper.

There is at least one survey covering the whole area, searchable as *A Survey on 3D Gaussian
Splatting*, arXiv:2401.03890. Useful for finding what exists, not for learning the material.

## Part 5: the rendering and browser half

Only needed if the project keeps the "runs in anyone's browser" goal, which is the part the spike
never tested.

- antimatter15's splat viewer, github.com/antimatter15/splat. It is one `main.js` of about 1200
  lines and it is readable. Clone it (already in `_webviewer/`) and read how it parses the file,
  sorts by depth, and issues the draw calls. This teaches more than any tutorial.
- WebGL fundamentals at webglfundamentals.org if you have never written a shader. WebGPU is the
  newer API, but learn WebGL concepts first; they transfer.
- The specific problems to understand: depth sorting a million primitives every frame without
  stalling, instanced rendering, and why the naive approach falls over on mobile.

Rough time: 15 to 20 hours if you have not done graphics programming before. This is the largest
single unknown in the whole plan, which is exactly why it should be tested early rather than late.

## Part 6: building CUDA extensions

You do not need to write CUDA kernels for this project, but the spike hit three bugs in this layer
and you should understand what was going on.

- PyTorch docs, *Custom C++ and CUDA Extensions*. Read enough to know what `torch.utils.cpp_extension`
  does, what JIT compilation means, and why the CUDA version has to match the one PyTorch was built
  against.
- NVIDIA's CUDA C++ Programming Guide, first two chapters, for the thread and block model. Skip the
  rest until you need it.

Rough time: 3 hours. Read `SPIKE_LOG.txt` alongside it, since the three bugs are worked examples.

## Part 7: adjacent, for later

These are the papers already sitting unread in `fun papers/`. They are not needed for the splat
project but they are the natural next step and they connect to it.

- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025, arXiv:2503.11651. Feed
  forward 3D reconstruction, no optimisation loop. Relevant because it could replace COLMAP in the
  pose estimation step.
- *MegaSaM*, arXiv:2412.04463. Structure and motion from casual video.
- Wang et al., *DUSt3R*, arXiv:2312.14132, and the follow-up *MASt3R*, arXiv:2406.09756. The line of
  work VGGT sits in. DUSt3R first, it is the more readable of the two.

## Suggested order if you only have weekends

| Weekend | What |
|---|---|
| 1 | Part 1, camera geometry. Nothing else makes sense without it. |
| 2 | Part 1 continued, plus the 3DGS project page video. |
| 3 | 3DGS paper, first pass. Do not worry about the maths yet. |
| 4 | Part 3, spherical harmonics, then re-read the 3DGS appendix. |
| 5 | 3DGS paper, second pass, now with the maths. Read gsplat source alongside. |
| 6 | Part 5, the viewer. Read `main.js`, get the truck scene rendering locally. |
| 7 | Part 4, compression. Aras's blog first, then one paper. |

That is roughly two months of weekends to go from no 3D vision to being able to hold a real
conversation about this work. The geometry in part 1 is the part that transfers everywhere else, so
it is worth the time even if the splat project never happens.

## How to tell it is working

You should be able to answer these without looking anything up:

1. What is in a camera intrinsics matrix and what happens to it when you halve the image resolution.
2. Why 3DGS stores scale and rotation separately instead of a covariance matrix.
3. What adaptive density control does and why training needs it.
4. What a spherical harmonic coefficient represents and why there are 45 of them per Gaussian.
5. Why sorting is the bottleneck in a splat renderer.
6. What `.splat` throws away compared to `.ply`.

If those are solid, you understand this work well enough to defend design decisions in an interview,
which is the actual goal.
