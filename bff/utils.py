"""Small request-parsing + file-resolution helpers shared across routers."""

from pathlib import Path

# Where bind-mounted data files (ui_themes.yaml, _configs/*) can live: the
# container workdir ``/app`` (where docker-compose mounts them), then the repo's
# ``_container_data`` (for local dev / tests where cwd is ``bff/``).
_DATA_BASE_DIRS: tuple[Path, ...] = (
    Path("."),
    Path(__file__).resolve().parents[1] / "_container_data",
)


def resolve_data_file(relpath: str) -> Path | None:
    """Return the first existing path for ``relpath`` across the known base dirs.

    ``relpath`` is given relative to ``_container_data`` (e.g. ``"ui_themes.yaml"``
    or ``"_configs/world_bank_download_config.json"``). Returns ``None`` when the
    file is not present in any base dir, so callers can surface a clean 503.
    """
    for base in _DATA_BASE_DIRS:
        candidate = base / relpath
        if candidate.is_file():
            return candidate
    return None


def parse_code_filter(value: str | None) -> list[str]:
    """Split a comma-separated code filter into a list of upper-trimmed codes.

    Treats ``None``, empty, and ``"ALL"`` (case-insensitive, in any position)
    as "no filter" and returns an empty list, mirroring the Streamlit app's
    ``_normalize_country_codes`` semantics.

    Args:
        value: Raw query-param string, e.g. ``"USA,DEU"`` or ``"ALL"``.

    Returns:
        List of codes, or ``[]`` for the unfiltered case.
    """
    if not value or not value.strip():
        return []
    codes = [part.strip() for part in value.split(",") if part.strip()]
    if not codes or any(code.upper() == "ALL" for code in codes):
        return []
    return codes
