import type { EChartsOption } from "echarts";

export interface TreeNode {
  name: string;
  value?: number;
  children?: TreeNode[];
  /** Extra hover detail line. */
  detail?: string;
}

/**
 * Sector → ticker treemap. Node colours are assigned by ECharts from the
 * registered theme colorway (so they follow the active palette).
 */
export function buildTreemapOption(nodes: TreeNode[], title = ""): EChartsOption {
  return {
    tooltip: {
      formatter: (params: unknown) => {
        const p = params as { name: string; value: number; data?: { detail?: string } };
        const detail = p.data?.detail ? `<br/>${p.data.detail}` : "";
        const value = Number.isFinite(p.value) ? p.value.toLocaleString() : "";
        return `<b>${p.name}</b>${detail}${value ? `<br/>${value}` : ""}`;
      },
    },
    series: [
      {
        type: "treemap",
        name: title,
        roam: false,
        nodeClick: false,
        breadcrumb: { show: false },
        label: { show: true, formatter: "{b}" },
        upperLabel: { show: true, height: 20 },
        levels: [
          { itemStyle: { borderColor: "transparent", borderWidth: 0, gapWidth: 2 } },
          { itemStyle: { gapWidth: 1 } },
        ],
        data: nodes,
      },
    ],
  };
}
