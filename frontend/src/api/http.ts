/**
 * Base fetch helpers. Everything goes through `/api`, which Vite (dev) and nginx
 * (prod) proxy to the BFF — so the browser always talks same-origin.
 */

const API_BASE = "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(response.status, `${response.status} ${detail}`);
  }
  return (await response.json()) as T;
}

export type QueryParams = Record<string, string | number | boolean | undefined | null>;

export async function getJson<T>(path: string, params?: QueryParams): Promise<T> {
  const url = new URL(API_BASE + path, window.location.origin);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }
  const response = await fetch(url.toString(), { headers: { Accept: "application/json" } });
  return parse<T>(response);
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return parse<T>(response);
}
