import { ExternalLink, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { ApiError } from "@/api/http";
import { useNewsBrowse, useNewsCollections, useNewsSearch } from "@/api/hooks";
import { buildWordcloudOption, computeWordFrequencies } from "@/charts/wordcloud";
import { EChart } from "@/components/charts/EChart";
import { EmbeddingMap } from "@/components/EmbeddingMap";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { useTheme } from "@/theme/useTheme";

/** News explorer — browse a Qdrant collection, word cloud, and semantic search. */
export function NewsPage() {
  const { config } = useTheme();
  const collectionsQuery = useNewsCollections();
  const collections = useMemo(
    () => collectionsQuery.data?.collections ?? [],
    [collectionsQuery.data],
  );

  const [collection, setCollection] = useState<string>("");
  useEffect(() => {
    if (!collection && collections.length) setCollection(collections[0]);
  }, [collection, collections]);

  const browseQuery = useNewsBrowse(collection || undefined, 200);
  const articles = useMemo(() => browseQuery.data ?? [], [browseQuery.data]);

  const [selectedId, setSelectedId] = useState<string>("");
  useEffect(() => {
    setSelectedId(articles[0]?.id ?? "");
  }, [articles]);
  const selected = articles.find((a) => a.id === selectedId);

  const wordcloudOption = useMemo(() => {
    const corpus = articles.map((a) => a.text).join(" ");
    const words = computeWordFrequencies(corpus, 150);
    return buildWordcloudOption(words, {
      colors: config.wordcloud.colors,
      background: config.wordcloud.background,
    });
  }, [articles, config]);

  // --- Semantic search -------------------------------------------------------
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const search = useNewsSearch();

  const runSearch = () => {
    if (query.trim()) search.mutate({ query: query.trim(), top_k: topK });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">News Explorer</h2>
        <p className="max-w-3xl text-muted-foreground">
          Browse a topic (Qdrant collection), inspect an article, see its word cloud, and run
          semantic search across the RAG corpus.
        </p>
      </div>

      {collectionsQuery.isLoading ? (
        <p className="text-muted-foreground">Loading collections…</p>
      ) : collections.length === 0 ? (
        <p className="text-muted-foreground">No news collections were found.</p>
      ) : (
        <>
          <div className="max-w-sm space-y-1">
            <label className="text-sm text-muted-foreground">News topic (collection)</label>
            <Select
              aria-label="News topic"
              value={collection}
              onChange={(event) => setCollection(event.target.value)}
            >
              {collections.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
          </div>

          <section className="space-y-2 rounded-lg border bg-card p-4">
            <h3 className="font-semibold">Topic word cloud</h3>
            <p className="text-xs text-muted-foreground">
              Built from {articles.length} browsed article(s).
            </p>
            <div className="h-80">
              {browseQuery.isLoading ? (
                <div className="grid h-full place-items-center text-sm text-muted-foreground">
                  Loading…
                </div>
              ) : (
                <EChart option={wordcloudOption} />
              )}
            </div>
          </section>

          <section className="space-y-3 rounded-lg border bg-card p-4">
            <h3 className="font-semibold">Article finder</h3>
            <Select
              aria-label="Select article"
              value={selectedId}
              onChange={(event) => setSelectedId(event.target.value)}
            >
              {articles.map((article) => (
                <option key={article.id} value={article.id}>
                  {article.title || article.id}
                  {article.published ? ` — ${article.published.slice(0, 10)}` : ""}
                </option>
              ))}
            </Select>
            {selected && (
              <div className="space-y-2">
                <h4 className="text-lg font-semibold">{selected.title || "Untitled"}</h4>
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  {selected.source && <span>Source: {selected.source}</span>}
                  {selected.topic && <span>· Topic: {selected.topic}</span>}
                  {selected.sentiment && <span>· Sentiment: {selected.sentiment}</span>}
                </div>
                {selected.url && (
                  <a
                    href={selected.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                  >
                    <ExternalLink className="h-3.5 w-3.5" /> Open original source
                  </a>
                )}
                <p className="max-h-80 overflow-auto whitespace-pre-wrap text-sm text-muted-foreground">
                  {selected.text || "No article text is available for this record."}
                </p>
              </div>
            )}
          </section>

          {collection && (
            <EmbeddingMap
              collection={collection}
              selectedId={selectedId}
              selectedTitle={selected?.title ?? ""}
            />
          )}

          <section className="space-y-3 rounded-lg border bg-card p-4">
            <h3 className="font-semibold">Semantic search</h3>
            <div className="flex flex-wrap items-end gap-2">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && runSearch()}
                placeholder="Search the news corpus…"
                className="h-9 flex-1 rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <div className="w-24 space-y-1">
                <label className="text-xs text-muted-foreground">Top K</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={topK}
                  onChange={(event) => setTopK(Math.min(50, Math.max(1, Number(event.target.value))))}
                  className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
                />
              </div>
              <Button onClick={runSearch} disabled={search.isPending || !query.trim()}>
                <Search className="mr-1 h-4 w-4" />
                {search.isPending ? "Searching…" : "Search"}
              </Button>
            </div>

            {search.isError && (
              <p className="text-sm text-negative">
                {search.error instanceof ApiError && search.error.status === 503
                  ? "Semantic search is disabled: the BFF has no OpenAI key configured."
                  : `Search failed: ${(search.error as Error).message}`}
              </p>
            )}
            {search.data?.message && (
              <p className="text-sm text-muted-foreground">{search.data.message}</p>
            )}
            {search.data && (
              <ul className="space-y-2">
                {search.data.articles.map((hit) => (
                  <li key={hit.id} className="rounded-md border p-3">
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-medium">{hit.title || "Untitled"}</span>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        score {hit.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="mt-1 line-clamp-3 text-sm text-muted-foreground">{hit.text}</p>
                    {hit.url && (
                      <a
                        href={hit.url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-1 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" /> source
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </div>
  );
}
