from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_policy_solver_facade_exports_core_api() -> None:
    from policy_solver import (
        ChunkPolicyConfig,
        build_candidate_bank_from_frames,
        build_chunk_policy_dataset,
        viterbi_select_candidates,
    )

    assert ChunkPolicyConfig.__name__ == "ChunkPolicyConfig"
    assert callable(build_candidate_bank_from_frames)
    assert callable(build_chunk_policy_dataset)
    assert callable(viterbi_select_candidates)


def test_policy_solver_train_cli_delegates_to_chunk_policy_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "policy_solver.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--chunk-size" in result.stdout
    assert "--step-predictions" in result.stdout


def test_policy_solver_readme_documents_pipeline_boundaries() -> None:
    readme = Path("policy_solver/README.md").read_text(encoding="utf-8")

    assert "candidates -> features -> models -> DP -> report" in readme
    assert "schema-safe" in readme
    assert "diagnostic-only" in readme
