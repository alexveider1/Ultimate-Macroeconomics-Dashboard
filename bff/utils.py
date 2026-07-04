"""Small request-parsing helpers shared across routers."""


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
