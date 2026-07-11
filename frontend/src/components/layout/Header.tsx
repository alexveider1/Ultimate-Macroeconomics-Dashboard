import { ThemeSwitcher } from "@/components/ThemeSwitcher";

export function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b bg-card px-6">
      <h1 className="truncate text-sm font-medium text-muted-foreground">
        Ultimate Macroeconomics Dashboard
      </h1>
      <ThemeSwitcher />
    </header>
  );
}
