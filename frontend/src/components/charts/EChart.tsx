import * as echarts from "echarts";
import { useEffect, useRef, type CSSProperties } from "react";

import { ECHARTS_THEME_NAME } from "@/theme/echartsTheme";
import { useTheme } from "@/theme/useTheme";

export interface EChartProps {
  option: echarts.EChartsOption;
  className?: string;
  style?: CSSProperties;
  /** Called with the instance after (re)creation — e.g. to grab a PNG for the LLM. */
  onReady?: (instance: echarts.ECharts) => void;
}

/**
 * Thin declarative wrapper around ECharts. The instance is (re)created whenever
 * the active theme changes so it re-reads the registered `app` theme (series
 * colours, axes, tooltip). A ResizeObserver keeps it fitted to its container;
 * callers give it a sized parent (e.g. `h-96`).
 */
export function EChart({ option, className, style, onReady }: EChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const { themeName } = useTheme();

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = echarts.init(el, ECHARTS_THEME_NAME, { renderer: "canvas" });
    chartRef.current = chart;
    onReady?.(chart);

    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(el);
    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
    // Re-create on theme change so the registered theme is re-read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [themeName]);

  useEffect(() => {
    chartRef.current?.setOption(option, true);
  }, [option]);

  return (
    <div ref={containerRef} className={className} style={{ width: "100%", height: "100%", ...style }} />
  );
}
