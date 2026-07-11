import { type ReactNode } from "react";

import { cn } from "@/lib/utils";

export interface DataTableColumn {
  key: string;
  header: string;
  align?: "left" | "right";
  className?: string;
}

export interface DataTableProps {
  columns: DataTableColumn[];
  rows: Record<string, ReactNode>[];
  /** Tailwind max-height for the scroll container (e.g. `max-h-96`). */
  maxHeightClass?: string;
  emptyMessage?: string;
}

/** Compact scrollable table with a sticky header. Cells are pre-rendered nodes. */
export function DataTable({
  columns,
  rows,
  maxHeightClass = "max-h-[28rem]",
  emptyMessage = "No rows.",
}: DataTableProps) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyMessage}</p>;
  }
  return (
    <div className={cn("overflow-auto rounded-md border", maxHeightClass)}>
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-muted/80 backdrop-blur">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "whitespace-nowrap px-3 py-2 font-medium text-muted-foreground",
                  col.align === "right" ? "text-right" : "text-left",
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-t hover:bg-accent/40">
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn(
                    "whitespace-nowrap px-3 py-1.5 tabular-nums",
                    col.align === "right" ? "text-right" : "text-left",
                    col.className,
                  )}
                >
                  {row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
