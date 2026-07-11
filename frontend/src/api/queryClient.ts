import { QueryClient } from "@tanstack/react-query";

/**
 * Shared query client. `staleTime` ~1h mirrors the Streamlit app's
 * `st.cache_data(ttl=3600)` on the same reads, so navigating between pages
 * doesn't re-hit the BFF for data that rarely changes.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 60 * 1000,
      gcTime: 2 * 60 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
