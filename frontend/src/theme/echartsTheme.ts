/**
 * Build + register an ECharts theme from a `ThemeConfig`. Every chart uses this
 * registered theme (name `app`), so series colours, axes, grid, tooltip and text
 * all come from the config — charts hard-code nothing.
 *
 * NOTE: imports the full `echarts` bundle for development velocity across many
 * chart types; switching to `echarts/core` + explicit `use()` registration is a
 * future bundle-size optimization.
 */

import * as echarts from "echarts";

import type { ThemeConfig } from "./types";

export const ECHARTS_THEME_NAME = "app";

function axisCommon(theme: ThemeConfig) {
  return {
    axisLine: { lineStyle: { color: theme.charts.axisLine } },
    axisTick: { lineStyle: { color: theme.charts.axisLine } },
    axisLabel: { color: theme.chrome.textMuted },
    splitLine: { lineStyle: { color: theme.charts.gridLine } },
    splitArea: { areaStyle: { color: ["transparent"] } },
  };
}

export function buildEchartsTheme(theme: ThemeConfig): object {
  const axis = axisCommon(theme);
  return {
    color: theme.series.colorway,
    backgroundColor: "transparent",
    textStyle: { color: theme.chrome.text, fontFamily: theme.fontFamily },
    title: {
      textStyle: { color: theme.chrome.text },
      subtextStyle: { color: theme.chrome.textMuted },
    },
    line: { itemStyle: { borderWidth: 2 }, lineStyle: { width: 2 }, symbolSize: 6 },
    categoryAxis: axis,
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    legend: { textStyle: { color: theme.chrome.text } },
    tooltip: {
      backgroundColor: theme.charts.tooltipBackground,
      borderColor: theme.chrome.border,
      textStyle: { color: theme.charts.tooltipText },
      axisPointer: {
        lineStyle: { color: theme.semantic.referenceLine },
        crossStyle: { color: theme.semantic.referenceLine },
      },
    },
    visualMap: { textStyle: { color: theme.chrome.textMuted } },
    dataZoom: {
      borderColor: theme.chrome.border,
      textStyle: { color: theme.chrome.textMuted },
    },
    geo: {
      itemStyle: { areaColor: theme.semantic.mapLand, borderColor: theme.semantic.mapCoastline },
    },
  };
}

/** (Re)register the `app` ECharts theme for the given config. */
export function registerEchartsTheme(theme: ThemeConfig): void {
  echarts.registerTheme(ECHARTS_THEME_NAME, buildEchartsTheme(theme));
}
