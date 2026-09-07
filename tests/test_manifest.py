import hashlib
from types import SimpleNamespace

from splatpipe import manifest as manifest_module
from splatpipe.config import RunConfig, TrainConfig
from splatpipe.manifest import ArtifactRecord, RunManifest, collect_versions


def test_artifact_record_hashes_the_file(tmp_path):
    payload = b"forty two" * 100
    target = tmp_path / "artifacts" / "scene.splat"
    target.parent.mkdir()
    target.write_bytes(payload)

    record = ArtifactRecord.of(target, relative_to=tmp_path)
    assert record.path == "artifacts/scene.splat"
    assert record.bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_start_captures_the_config_and_its_digest():
    cfg = RunConfig(name="tiny", train=TrainConfig(max_steps=50))
    manifest = RunManifest.start(cfg)
    assert manifest.name == "tiny"
    assert manifest.config_digest == cfg.digest()
    assert manifest.config["train"]["max_steps"] == 50
    assert manifest.created_utc.endswith("Z")


def test_round_trips_through_json(tmp_path):
    manifest = RunManifest.start(RunConfig(name="tiny"))
    manifest.timings["train"] = 12.5
    manifest.metrics["psnr"] = 24.4
    manifest.artifacts.append(ArtifactRecord(path="artifacts/scene.splat", bytes=32, sha256="ab"))

    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert RunManifest.read(path) == manifest


def test_versions_include_what_would_change_a_result():
    versions = collect_versions()
    assert versions["python"].startswith("3.11")
    for key in ("numpy", "splatpipe", "platform"):
        assert versions[key]


def test_git_version_marks_an_uncommitted_tree(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=" M src/splatpipe/manifest.py\n"),
        ]
    )
    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )
    assert manifest_module._git_commit() == "abc123-dirty"
