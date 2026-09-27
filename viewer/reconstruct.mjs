// Turns a decoded container into render-ready arrays. DOM-free so Node can
// check it against Python's unpack_scene.

const SH_C0 = 0.28209479177387814;
const REQUIRED = ['means', 'scales', 'quats', 'opacities', 'sh0'];

// Same arithmetic as compress/quantize.py:dequantize_affine, rounded to
// float32 as Python's cloud is before any activation.
function dequantize(block) {
  const { values, bits, mins, maxs } = block;
  const components = mins.length;
  const levels = 2 ** bits - 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const c = i % components;
    out[i] = (values[i] / levels) * (maxs[c] - mins[c]) + mins[c];
  }
  return out;
}

export function reconstruct(decoded) {
  const { count, sh_degree: shDegree, blocks } = decoded;
  const missing = REQUIRED.filter((name) => !(name in blocks));
  if (missing.length) throw new Error(`container is missing block(s): ${missing.join(', ')}`);
  if (shDegree !== 0 && shDegree !== 3) {
    throw new Error(`unsupported sh_degree ${shDegree}, expected 0 or 3`);
  }

  const means = dequantize(blocks.means);
  for (let i = 0; i < means.length; i += 1) {
    means[i] = Math.sign(means[i]) * Math.expm1(Math.abs(means[i]));
  }

  const scales = dequantize(blocks.scales);
  for (let i = 0; i < scales.length; i += 1) scales[i] = Math.exp(scales[i]);

  const quats = dequantize(blocks.quats);
  for (let o = 0; o < quats.length; o += 4) {
    const norm = Math.hypot(quats[o], quats[o + 1], quats[o + 2], quats[o + 3]);
    if (norm === 0) {
      // gsplat normalises a zero quaternion to zero, whose rotation matrix is the identity.
      quats[o] = 1;
      continue;
    }
    for (let k = 0; k < 4; k += 1) quats[o + k] /= norm;
  }

  const opacities = dequantize(blocks.opacities);
  for (let i = 0; i < opacities.length; i += 1) opacities[i] = 1 / (1 + Math.exp(-opacities[i]));

  const colors = dequantize(blocks.sh0);
  for (let i = 0; i < colors.length; i += 1) colors[i] = 0.5 + SH_C0 * colors[i];

  let codebook = null;
  let codebookRows = 0;
  let labels = null;
  if ('sh_codebook' in blocks && 'sh_labels' in blocks) {
    codebook = dequantize(blocks.sh_codebook);
    codebookRows = blocks.sh_codebook.shape[0];
    labels = blocks.sh_labels.values;
    for (let i = 0; i < labels.length; i += 1) {
      if (labels[i] >= codebookRows) {
        throw new Error(`sh label ${labels[i]} exceeds the ${codebookRows}-entry codebook`);
      }
    }
  } else if ('shN' in blocks) {
    throw new Error('container stores uncompressed shN; the viewer reads only an sh codebook');
  }

  return { count, means, scales, quats, opacities, colors, codebook, codebookRows, labels };
}
