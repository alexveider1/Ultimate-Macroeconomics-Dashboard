// Ambient module declarations for packages that ship no TypeScript types.

/** ECharts word-cloud extension — imported for its side effect (registers `wordCloud`). */
declare module "echarts-wordcloud";

/** ECharts GL extension — imported for its side effect (registers `scatter3D`/`grid3D`). */
declare module "echarts-gl";

/** Pre-bundled Plotly build used via the react-plotly.js factory on the chat page only. */
declare module "plotly.js-dist-min" {
  const Plotly: unknown;
  export default Plotly;
}

/** The react-plotly.js factory: builds a `<Plot>` component around a Plotly instance. */
declare module "react-plotly.js/factory" {
  import type * as React from "react";
  import type { PlotParams } from "react-plotly.js";

  export default function createPlotlyComponent(
    plotly: unknown,
  ): React.ComponentType<PlotParams>;
}
