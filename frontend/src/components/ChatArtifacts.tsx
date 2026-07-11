import { Download } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";

import { DataTable } from "@/components/DataTable";
import { PlotlyArtifact } from "@/components/PlotlyArtifact";

const PREVIEW_LIMIT = 100;

interface TableArtifact {
  rows?: unknown[];
  columns?: string[];
  row_count?: number;
  truncated?: boolean;
}

function toCsv(columns: string[], rows: Record<string, unknown>[]): string {
  const escape = (value: unknown) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const header = columns.map(escape).join(",");
  const body = rows.map((row) => columns.map((col) => escape(row[col])).join(",")).join("\n");
  return `${header}\n${body}`;
}

/** Renders the plot + data-table block emitted by the agent's `final` event. */
export function ChatArtifacts({ artifacts }: { artifacts: Record<string, unknown> }) {
  const [showTable, setShowTable] = useState(false);

  const plot = artifacts.latest_plotly as { figure_json?: string; title?: string } | undefined;
  const table = (artifacts.latest_table ?? artifacts.latest_data) as TableArtifact | undefined;

  const tableView = useMemo(() => {
    const rows = (table?.rows ?? []).filter((r): r is Record<string, unknown> =>
      Boolean(r) && typeof r === "object",
    );
    if (rows.length === 0) return null;
    const columns = table?.columns?.length
      ? table.columns
      : [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const preview = rows.slice(0, PREVIEW_LIMIT).map((row) => {
      const cell: Record<string, ReactNode> = {};
      for (const col of columns) {
        const value = row[col];
        cell[col] = value === null || value === undefined ? "" : String(value);
      }
      return cell;
    });
    return {
      columns: columns.map((c) => ({ key: c, header: c })),
      preview,
      rowCount: table?.row_count ?? rows.length,
      previewCount: preview.length,
      csv: toCsv(columns, rows),
      truncated: Boolean(table?.truncated),
    };
  }, [table]);

  if (!plot?.figure_json && !tableView) return null;

  return (
    <div className="mt-2 space-y-2">
      {plot?.figure_json && <PlotlyArtifact figureJson={plot.figure_json} title={plot.title} />}
      {tableView && (
        <div>
          <button
            type="button"
            onClick={() => setShowTable((value) => !value)}
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            {showTable ? "Hide data table" : "Show data table"}
          </button>
          {showTable && (
            <div className="mt-2 space-y-2">
              <DataTable columns={tableView.columns} rows={tableView.preview} />
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  {tableView.rowCount > tableView.previewCount
                    ? `Showing the first ${tableView.previewCount} of ${tableView.rowCount} row(s).`
                    : `${tableView.rowCount} row(s).`}
                </span>
                <a
                  href={`data:text/csv;charset=utf-8,${encodeURIComponent(tableView.csv)}`}
                  download="agent_query_result.csv"
                  className="inline-flex items-center gap-1 text-primary hover:underline"
                >
                  <Download className="h-3 w-3" /> Download CSV
                </a>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
