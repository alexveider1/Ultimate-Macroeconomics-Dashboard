import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";

export interface ForecastSettings {
  model: string;
  modelParams: Record<string, number>;
  alpha: number;
  pointsToUse: number;
  pointsToPredict: number;
}

/** One numeric hyperparameter widget spec (mirrors `_render_model_param_inputs`). */
interface ParamSpec {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

/** Per-model hyperparameter widgets, matching the Streamlit GraphBox exactly. */
const MODEL_PARAMS: Record<string, ParamSpec[]> = {
  arima: [
    { key: "p", label: "AR order (p)", min: 0, max: 10, step: 1, default: 1 },
    { key: "d", label: "Diff order (d)", min: 0, max: 3, step: 1, default: 1 },
    { key: "q", label: "MA order (q)", min: 0, max: 10, step: 1, default: 1 },
  ],
  sarima: [
    { key: "p", label: "AR order (p)", min: 0, max: 10, step: 1, default: 1 },
    { key: "d", label: "Diff order (d)", min: 0, max: 3, step: 1, default: 1 },
    { key: "q", label: "MA order (q)", min: 0, max: 10, step: 1, default: 1 },
    { key: "P", label: "Seasonal AR (P)", min: 0, max: 5, step: 1, default: 0 },
    { key: "D", label: "Seasonal diff (D)", min: 0, max: 2, step: 1, default: 0 },
    { key: "Q", label: "Seasonal MA (Q)", min: 0, max: 5, step: 1, default: 0 },
    { key: "s", label: "Seasonal period (s)", min: 1, max: 365, step: 1, default: 12 },
  ],
  moving_average: [{ key: "window", label: "Window", min: 1, max: 100, step: 1, default: 5 }],
  xgboost: [
    { key: "lags", label: "Lags", min: 1, max: 60, step: 1, default: 5 },
    { key: "n_estimators", label: "Estimators", min: 50, max: 1000, step: 50, default: 200 },
    { key: "max_depth", label: "Max depth", min: 1, max: 12, step: 1, default: 3 },
    { key: "learning_rate", label: "Learning rate", min: 0.005, max: 0.5, step: 0.005, default: 0.05 },
  ],
};

/** Fallback model list matching the Streamlit dropdown order (if /forecast/models is down). */
const FALLBACK_MODELS = [
  "prophet",
  "auto_arima",
  "arima",
  "sarima",
  "moving_average",
  "xgboost",
  "chronos",
];

interface ForecastControlsProps {
  models: string[] | undefined;
  running: boolean;
  hasForecast: boolean;
  onRun: (settings: ForecastSettings) => void;
  onClear: () => void;
}

function NumberField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-sm">
      <span className="text-muted-foreground">{spec.label}</span>
      <input
        type="number"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 w-24 rounded-md border border-input bg-background px-2 text-right text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      />
    </label>
  );
}

/**
 * Forecasting controls: model dropdown → per-model hyperparameter widgets, alpha,
 * lookback + horizon, and Run / Clear. Self-contained state; emits a
 * `ForecastSettings` on Run. No colours are set here — purely form controls.
 */
export function ForecastControls({
  models,
  running,
  hasForecast,
  onRun,
  onClear,
}: ForecastControlsProps) {
  const modelList = models && models.length > 0 ? models : FALLBACK_MODELS;
  const [model, setModel] = useState(modelList.includes("prophet") ? "prophet" : modelList[0]);
  const [paramValues, setParamValues] = useState<Record<string, number>>({});
  const [alpha, setAlpha] = useState(0.05);
  const [pointsToUse, setPointsToUse] = useState(50);
  const [pointsToPredict, setPointsToPredict] = useState(10);

  const specs = useMemo(() => MODEL_PARAMS[model] ?? [], [model]);

  const currentParams = useMemo(() => {
    const out: Record<string, number> = {};
    for (const spec of specs) out[spec.key] = paramValues[`${model}:${spec.key}`] ?? spec.default;
    return out;
  }, [specs, paramValues, model]);

  const setParam = (key: string, value: number) =>
    setParamValues((prev) => ({ ...prev, [`${model}:${key}`]: value }));

  return (
    <div className="space-y-2">
      <div className="space-y-1">
        <Label>Forecast model</Label>
        <Select
          aria-label="Forecast model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {modelList.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </Select>
      </div>

      {specs.map((spec) => (
        <NumberField
          key={spec.key}
          spec={spec}
          value={currentParams[spec.key]}
          onChange={(v) => setParam(spec.key, v)}
        />
      ))}

      <NumberField
        spec={{ key: "alpha", label: "Alpha (CI)", min: 0.01, max: 0.2, step: 0.01, default: 0.05 }}
        value={alpha}
        onChange={setAlpha}
      />
      <NumberField
        spec={{ key: "n_prev", label: "Points to use", min: 6, max: 500, step: 1, default: 50 }}
        value={pointsToUse}
        onChange={setPointsToUse}
      />
      <NumberField
        spec={{
          key: "n_predict",
          label: "Points to predict",
          min: 1,
          max: pointsToUse,
          step: 1,
          default: 10,
        }}
        value={pointsToPredict}
        onChange={setPointsToPredict}
      />

      <div className="flex gap-2 pt-1">
        <Button
          size="sm"
          className="flex-1"
          disabled={running}
          onClick={() =>
            onRun({
              model,
              modelParams: currentParams,
              alpha,
              pointsToUse,
              pointsToPredict: Math.min(pointsToPredict, pointsToUse),
            })
          }
        >
          {running ? "Running…" : "Run forecast"}
        </Button>
        {hasForecast && (
          <Button size="sm" variant="outline" onClick={onClear} disabled={running}>
            Clear
          </Button>
        )}
      </div>
    </div>
  );
}
