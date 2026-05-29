from pathlib import Path


def test_chunk_policy_make_target_forwards_loss_target_overrides() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "--cost-target $(CHUNK_POLICY_COST_TARGET)" in makefile
    assert "--ranker-target $(CHUNK_POLICY_RANKER_TARGET)" in makefile
