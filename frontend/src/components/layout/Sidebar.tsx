import { NavLink } from "react-router-dom";

import { NAV } from "@/config/nav";
import { cn } from "@/lib/utils";

export function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r bg-card md:flex">
      <div className="flex h-14 shrink-0 items-center gap-2 border-b px-4 font-semibold">
        <span aria-hidden>🌍</span>
        <span>Macro Dashboard</span>
      </div>
      <nav className="flex-1 space-y-4 overflow-y-auto p-3">
        {NAV.map((group) => (
          <div key={group.label}>
            <p className="px-2 pb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {group.label}
            </p>
            <ul className="space-y-0.5">
              {group.items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    className={({ isActive }) =>
                      cn(
                        "block rounded-md px-2 py-1.5 text-sm text-foreground/80 transition-colors hover:bg-accent hover:text-accent-foreground",
                        isActive && "bg-accent font-medium text-accent-foreground",
                      )
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}
