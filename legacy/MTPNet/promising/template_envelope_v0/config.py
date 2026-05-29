from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def parse_float_grid(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Parse comma grids or start:step:end grids."""
    if isinstance(value, tuple):
        return tuple(float(x) for x in value)
    if isinstance(value, list):
        return tuple(float(x) for x in value)
    raw = str(value).strip()
    if not raw:
        return tuple()
    if ":" in raw and "," not in raw:
        parts = [float(part) for part in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("colon grids must be start:step:end")
        start, step, end = parts
        if step == 0:
            raise ValueError("grid step must be non-zero")
        values: list[float] = []
        current = start
        if step > 0:
            while current <= end + 1e-9:
                values.append(round(current, 10))
                current += step
        else:
            while current >= end - 1e-9:
                values.append(round(current, 10))
                current += step
        return tuple(values)
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


VALID_COORD_SOURCES: tuple[str, ...] = (
    "z_aligned",
    "bridge_slope_z_capped",
    "bridge_slope",
    "bridge",
    "md_aligned",
)


@dataclass(frozen=True)
class TemplateEnvelopeConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/template_envelope_v0")
    rows_per_step: int = 32
    k_wells: int = -1
    seed: int = 42
    # Wider scale grid: the late hengck23 idea uses TW_GR(a*tvt + offset) with
    # potentially large stretch. ±15% (old default) is too narrow to capture
    # geological undulation across the hidden region.
    scale_grid: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5)
    offset_grid: tuple[float, ...] = (
        -160.0,
        -140.0,
        -120.0,
        -100.0,
        -80.0,
        -60.0,
        -40.0,
        -20.0,
        0.0,
        20.0,
        40.0,
        60.0,
        80.0,
        100.0,
        120.0,
        140.0,
        160.0,
    )
    top_n_templates: int = 16
    envelope_low_quantile: float = 0.0
    envelope_high_quantile: float = 1.0
    variants: tuple[str, ...] = ("normal_GR", "shuffled_hidden_GR", "zero_hidden_GR")
    progress_every: int = 100
    # Coordinate proxy for the template path. The original `bridge` proxy
    # clips beyond known TVT_input endpoints and collapses to a constant on
    # the hidden region in real ROGII wells (TVT_input is heel-only). The
    # naive `bridge_slope` proxy fits a line through known TVT_input rows and
    # extrapolates linearly forever, which is wrong for horizontal wells
    # (once the well lies down, TVT barely changes).
    # `z_aligned` uses -Z (TVD) shifted to match the mean known TVT_input;
    # Z is observed everywhere and naturally flattens through the horizontal
    # section. Used as the default.
    # `bridge_slope_z_capped` follows bridge_slope only while Z is changing
    # and then locks the coord to the last known TVT once Z plateaus.
    coord_source: str = "z_aligned"
    # Minimum scoring threshold for a template to be eligible for selection.
    # The score is a weighted z-correlation. 0.1 is a gentle floor that keeps
    # only mildly-positive correlations; combined with `top_n_templates` it
    # acts as a soft cap. Empirically on ROGII data top-scores rarely exceed
    # 0.5, so a stricter floor starves most wells of any envelope at all.
    min_template_score: float = 0.1
    # Require selected templates to also beat the median shuffled-variant
    # score on the same well (defense against pure location/bridge signal).
    require_beats_shuffled: bool = False

