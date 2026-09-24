"""The viewer's reconstruction must agree with Python's unpack_scene, or the
browser renders a different scene from the one the benchmark scored."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from splatpipe.formats.container import ShVq, pack_scene, read_container, unpack_scene, write_container
from tests.test_codecs import a_cloud

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
SH_C0 = 0.28209479177387814
ULP4 = 4 * 2.0**-23

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")

RECONSTRUCT_FILE = """
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const [viewer, input, output] = process.argv.slice(1);
const { decodeContainer } = await import(pathToFileURL(`${viewer}/decode_container.mjs`));
const { reconstruct } = await import(pathToFileURL(`${viewer}/reconstruct.mjs`));
const file = await readFile(input);
const scene = reconstruct(await decodeContainer(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)));
const out = {};
for (const [key, value] of Object.entries(scene)) out[key] = ArrayBuffer.isView(value) ? Array.from(value) : value;
await writeFile(output, JSON.stringify(out));
"""

# Builds a decoded container by hand, for faults the Python encoder refuses to write.
RECONSTRUCT_LITERAL = """
import { pathToFileURL } from 'node:url';
const [viewer, literal] = process.argv.slice(1);
const { reconstruct } = await import(pathToFileURL(`${viewer}/reconstruct.mjs`));
const decoded = JSON.parse(literal);
for (const block of Object.values(decoded.blocks)) {
  block.values = block.bits > 8 ? Uint16Array.from(block.values) : Uint8Array.from(block.values);
}
const scene = reconstruct(decoded);
const out = {};
for (const [key, value] of Object.entries(scene)) out[key] = ArrayBuffer.isView(value) ? Array.from(value) : value;
console.log(JSON.stringify(out));
"""


def _node(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "--input-type=module", "-e", script, str(VIEWER), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def a_vq_scene(n: int = 300, clusters: int = 16, label_dtype=np.uint16):
    rng = np.random.default_rng(1)
    sh_vq = ShVq(
        codebook=rng.integers(0, 64, (clusters, 45), dtype=np.uint8),
        labels=rng.integers(0, clusters, n).astype(label_dtype),
        mins=rng.uniform(-0.3, -0.1, 45).astype(np.float32),
        maxs=rng.uniform(0.1, 0.3, 45).astype(np.float32),
        bits=6,
    )
    return pack_scene(a_cloud(n), sh_vq=sh_vq, order="morton")


def _assert_close(actual, expected, name):
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32).reshape(expected.shape)
    atol = ULP4 * float(np.max(np.abs(expected), initial=0.0))
    np.testing.assert_allclose(actual, expected, rtol=ULP4, atol=atol, err_msg=name)


def _unit_quats(quats: np.ndarray) -> np.ndarray:
    q = quats.astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (len(q), 1))
    return np.where(norm > 0, q / np.where(norm > 0, norm, 1.0), identity)


@pytest.mark.parametrize("label_dtype", (np.uint8, np.uint16))
def test_reconstruction_matches_unpack_scene(tmp_path, label_dtype):
    path = tmp_path / "scene.splatc"
    write_container(a_vq_scene(label_dtype=label_dtype), path)
    stored = read_container(path)
    cloud = unpack_scene(stored)

    output = tmp_path / "scene.json"
    result = _node(RECONSTRUCT_FILE, str(path), str(output))
    assert result.returncode == 0, result.stderr
    js = json.loads(output.read_text(encoding="utf-8"))

    assert js["count"] == len(cloud)
    _assert_close(js["means"], cloud.means, "means")
    _assert_close(js["scales"], np.exp(cloud.scales.astype(np.float64)), "scales")
    _assert_close(js["quats"], _unit_quats(cloud.quats), "quats")
    _assert_close(js["opacities"], 1 / (1 + np.exp(-cloud.opacities.astype(np.float64))), "opacities")
    _assert_close(js["colors"], 0.5 + SH_C0 * cloud.sh0.astype(np.float64), "colors")

    codebook = stored.fields["sh_codebook"].dequantize()
    labels = stored.fields["sh_labels"].values
    assert js["codebookRows"] == len(codebook)
    _assert_close(js["codebook"], codebook, "codebook")
    np.testing.assert_array_equal(js["labels"], labels)
    # The shader reads coefficient k of a row as floats 3k..3k+2.
    np.testing.assert_array_equal(codebook[labels].reshape(len(labels), 15, 3), cloud.shN)


def _literal(**overrides) -> str:
    block = lambda values, bits, components, shape: {
        "values": values, "bits": bits, "shape": shape,
        "mins": [0.0] * components, "maxs": [1.0] * components,
    }
    decoded = {
        "count": 1,
        "sh_degree": 3,
        "order": "none",
        "blocks": {
            "means": block([0, 0, 0], 16, 3, [1, 3]),
            "scales": block([0, 0, 0], 8, 3, [1, 3]),
            "quats": block([0, 0, 0, 0], 8, 4, [1, 4]),
            "opacities": block([0], 8, 1, [1]),
            "sh0": block([0, 0, 0], 8, 3, [1, 3]),
            "sh_codebook": block([0] * 90, 6, 45, [2, 45]),
            "sh_labels": block([1], 16, 1, [1]),
        },
    }
    for key, value in overrides.items():
        if key == "drop":
            for name in value:
                del decoded["blocks"][name]
        elif key == "sh_degree":
            decoded["sh_degree"] = value
        else:
            decoded["blocks"][key] = value
    return json.dumps(decoded)


def test_a_zero_quaternion_becomes_the_identity():
    result = _node(RECONSTRUCT_LITERAL, _literal())
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["quats"] == [1.0, 0.0, 0.0, 0.0]


def test_a_container_without_sh_blocks_renders_dc_only():
    result = _node(RECONSTRUCT_LITERAL, _literal(drop=["sh_codebook", "sh_labels"], sh_degree=0))
    assert result.returncode == 0, result.stderr
    scene = json.loads(result.stdout)
    assert scene["codebook"] is None
    assert scene["labels"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"drop": ["scales"]}, "missing block(s): scales"),
        ({"sh_degree": 1}, "unsupported sh_degree 1"),
        ({"sh_labels": {"values": [2], "bits": 16, "shape": [1], "mins": [0.0], "maxs": [0.0]}}, "exceeds the 2-entry codebook"),
        ({"drop": ["sh_codebook", "sh_labels"], "shN": {"values": [0] * 45, "bits": 8, "shape": [1, 15, 3], "mins": [0.0] * 45, "maxs": [1.0] * 45}}, "uncompressed shN"),
    ),
)
def test_reconstruction_refuses_what_it_cannot_render(overrides, message):
    result = _node(RECONSTRUCT_LITERAL, _literal(**overrides))
    assert result.returncode != 0
    assert message in result.stderr
