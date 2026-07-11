import { Card } from "@/components/ui/card";

export interface MetricProps {
  label: string;
  value: string;
  caption?: string;
}

/** Small stat tile (label · big value · optional caption). */
export function Metric({ label, value, caption }: MetricProps) {
  return (
    <Card className="p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="truncate text-xl font-semibold tabular-nums">{value}</p>
      {caption && <p className="truncate text-xs text-muted-foreground">{caption}</p>}
    </Card>
  );
}
