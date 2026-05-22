from pathlib import Path

from mtpnet.codebase_bundle import build_bundle, collect_codebase_files


def test_collect_codebase_files_excludes_generated_dirs(tmp_path: Path) -> None:
    source_dir = "res" + "ources"
    research_file = "research" + "_101.md"
    (tmp_path / "mtpnet").mkdir()
    (tmp_path / "mtpnet" / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mtp.yml").write_text("run:\n  name: unit\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "well.csv").write_text("large,data\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / source_dir).mkdir()
    (tmp_path / source_dir / "paper.pdf").write_bytes(b"%PDF")
    (tmp_path / source_dir / "webinar.txt").write_text("transcript\n", encoding="utf-8")
    (tmp_path / research_file).write_text("# research\n", encoding="utf-8")
    (tmp_path / "docs" / "superpowers").mkdir(parents=True)
    (tmp_path / "docs" / "superpowers" / "plan.md").write_text(
        "private planning\n", encoding="utf-8"
    )

    files = [path.as_posix() for path in collect_codebase_files(tmp_path)]

    assert files == [
        "configs/mtp.yml",
        "mtpnet/model.py",
    ]


def test_build_bundle_contains_fenced_relative_file_contents(tmp_path: Path) -> None:
    (tmp_path / "mtpnet").mkdir()
    (tmp_path / "mtpnet" / "train.py").write_text("print('train')\n", encoding="utf-8")
    bundle = build_bundle(tmp_path)

    assert "# MTPNet Codebase Bundle" in bundle
    assert "File count: 1" in bundle
    assert "## mtpnet/train.py" in bundle
    assert "```python\nprint('train')\n```" in bundle
