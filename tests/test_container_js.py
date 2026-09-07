"""The format is a contract with a JavaScript viewer, so a JavaScript decoder
has to agree with the Python one byte for byte. Node is skipped rather than
failed when absent so the fast tier stays runnable on a bare checkout."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from splatpipe.formats.container import pack_scene, write_container
from tests.test_codecs import a_cloud

DECODER = Path(__file__).resolve().parents[1] / "scripts" / "decode_container.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not on PATH"
)


def _decode_with_node(path: Path, tmp_path: Path) -> dict:
    output = tmp_path / "decoded.json"
    result = subprocess.run(
        ["node", str(DECODER), str(path), str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("codec", ("raw", "deflate"))
def test_node_recovers_every_block_exactly(tmp_path, codec):
    scene = pack_scene(a_cloud(300), order="morton")
    path = tmp_path / "scene.splatc"
    write_container(scene, path, codecs={name: codec for name in scene.fields})

    decoded = _decode_with_node(path, tmp_path)
    assert decoded["count"] == scene.count
    assert decoded["sh_degree"] == scene.sh_degree
    assert decoded["order"] == "morton"
    for name, field in scene.fields.items():
        actual = np.asarray(decoded["blocks"][name]["values"], dtype=field.values.dtype)
        np.testing.assert_array_equal(
            actual, np.ascontiguousarray(field.values).reshape(-1), err_msg=name
        )


def test_node_refuses_a_tampered_block(tmp_path):
    scene = pack_scene(a_cloud(128), order="none")
    path = tmp_path / "scene.splatc"
    write_container(scene, path, codecs={name: "deflate" for name in scene.fields})
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))

    result = subprocess.run(
        ["node", str(DECODER), str(path), str(tmp_path / "out.json")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "checksum" in result.stderr.lower()
