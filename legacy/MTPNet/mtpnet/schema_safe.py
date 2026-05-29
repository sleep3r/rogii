from __future__ import annotations

FORBIDDEN_INFERENCE_COLUMNS: frozenset[str] = frozenset(
    {
        "TVT",
        "Geology",
        "ANCC",
        "ASTNU",
        "ASTNL",
        "EGFDU",
        "EGFDL",
        "BUDA",
    }
)

FORMATION_COLUMNS: tuple[str, ...] = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)


def forbidden_inference_columns(columns: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted({str(column) for column in columns}.intersection(FORBIDDEN_INFERENCE_COLUMNS))


def assert_schema_safe_columns(
    columns: list[str] | tuple[str, ...] | set[str],
    *,
    context: str = "feature builder",
) -> None:
    forbidden = forbidden_inference_columns(columns)
    if forbidden:
        raise ValueError(
            f"{context} contains forbidden train-only inference columns: {forbidden}"
        )
