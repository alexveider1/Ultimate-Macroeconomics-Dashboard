import type { EChartsOption } from "echarts";
// Side-effect import: registers the `wordCloud` series type on the shared echarts.
import "echarts-wordcloud";

export interface WordFrequency {
  name: string;
  value: number;
}

/** A compact English stop-word set so the cloud surfaces meaningful terms. */
const STOP_WORDS = new Set(
  (
    "the a an and or but if then else of to in on for with without within from by at as is are was " +
    "were be been being it its this that these those he she they them his her their our your you we " +
    "i me my mine us not no nor so than too very can will just don should now also into over under " +
    "about after before between out up down off again more most other some such only own same each " +
    "which who whom what when where why how all any both few has have had do does did done s t re ve " +
    "ll d m o re said say says according would could may might must new one two"
  ).split(" "),
);

/** Tokenise `text` and return the `maxWords` most frequent non-stop words. */
export function computeWordFrequencies(text: string, maxWords = 150): WordFrequency[] {
  const counts = new Map<string, number>();
  const tokens = text.toLowerCase().match(/[a-z][a-z'-]{2,}/g) ?? [];
  for (const raw of tokens) {
    const word = raw.replace(/^['-]+|['-]+$/g, "");
    if (word.length < 3 || STOP_WORDS.has(word)) continue;
    counts.set(word, (counts.get(word) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, maxWords);
}

export interface WordcloudOptions {
  /** Colour ramp cycled per word (theme `wordcloud.colors`). */
  colors: string[];
  background: string;
}

/**
 * Word cloud of term frequencies. Colours are drawn from the theme's explicit
 * `wordcloud.colors` ramp (no matplotlib colormap). The `wordCloud` series type
 * isn't in echarts' core typings, so the option is built loosely and cast.
 */
export function buildWordcloudOption(
  words: WordFrequency[],
  { colors, background }: WordcloudOptions,
): EChartsOption {
  // echarts-wordcloud calls `textStyle.color` with no args per word, so cycle a
  // closure counter through the theme ramp rather than relying on a data index.
  let colorIndex = 0;
  const series = {
    type: "wordCloud",
    shape: "circle",
    gridSize: 8,
    sizeRange: [12, 60],
    rotationRange: [-45, 45],
    rotationStep: 15,
    drawOutOfBound: false,
    textStyle: {
      color: () => colors[colorIndex++ % colors.length],
    },
    emphasis: { textStyle: { fontWeight: "bold" } },
    data: words,
  };
  return {
    backgroundColor: background,
    tooltip: { show: true, formatter: (p: unknown) => (p as { name: string }).name },
    series: [series],
  } as unknown as EChartsOption;
}
