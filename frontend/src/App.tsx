import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";

import { queryClient } from "@/api/queryClient";
import { router } from "@/router";
import { ThemeProvider } from "@/theme/ThemeProvider";

/**
 * Provider stack: QueryClient wraps ThemeProvider (which fetches the theme via a
 * query and blocks until it's applied), which wraps the router.
 */
export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <RouterProvider router={router} />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
