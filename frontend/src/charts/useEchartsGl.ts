import { useEffect, useState } from "react";

let glPromise: Promise<unknown> | null = null;

/**
 * Lazily side-effect-import `echarts-gl` (which registers the `scatter3D` /
 * `grid3D` types onto the shared echarts instance). It's ~1 MB, so it's only
 * pulled in when a page actually needs a 3D chart. Returns `true` once loaded.
 */
export function useEchartsGl(enabled: boolean): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (!enabled || ready) return;
    let cancelled = false;
    glPromise ??= import("echarts-gl");
    void glPromise.then(() => {
      if (!cancelled) setReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, [enabled, ready]);
  return ready;
}
