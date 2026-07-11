"""Smoke tests for app theming helpers."""

from __future__ import annotations

import pytest

from core.theming import (
    get_active_theme,
    get_color,
    get_colorway,
    get_diverging_colorscale,
    get_sequential_colorscale,
    register_plotly_template,
)


def test_active_theme_has_semantic_palette() -> None:
    theme = get_active_theme()
    assert isinstance(theme, dict)
    semantic = theme.get("semantic") or {}
    assert semantic, "active theme should expose a non-empty 'semantic' palette"


def test_get_color_raises_on_unknown_token() -> None:
    with pytest.raises(KeyError):
        get_color("this_token_does_not_exist")


def test_colorway_is_nonempty_list_of_hex_strings() -> None:
    colorway = get_colorway()
    assert isinstance(colorway, list) and colorway
    assert all(isinstance(c, str) and c.startswith("#") for c in colorway)


def test_diverging_and_sequential_colorscales_have_expected_shape() -> None:
    div = get_diverging_colorscale()
    seq = get_sequential_colorscale()
    assert [stop[0] for stop in div] == [0.0, 0.5, 1.0]
    assert [stop[0] for stop in seq] == [0.0, 1.0]


def test_register_plotly_template_does_not_raise() -> None:
    register_plotly_template()
