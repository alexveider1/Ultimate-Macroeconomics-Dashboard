/**
 * Sidebar navigation — mirrors the Streamlit `app.py` nav groups. Routes are
 * stable slugs; the Dashboard group's pages become config-driven `<IndicatorPage>`
 * instances in the next milestone. The Settings/monitoring pages are intentionally
 * excluded from the rewrite.
 */

export interface NavItem {
  label: string;
  to: string;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const NAV: NavGroup[] = [
  {
    label: "Dashboard",
    items: [
      { label: "General Economics", to: "/dashboard/general-economics" },
      { label: "Economy Structure", to: "/dashboard/economy-structure" },
      { label: "Finance & Monetary", to: "/dashboard/finance-monetary" },
      { label: "Trade & External", to: "/dashboard/trade" },
      { label: "Demography", to: "/dashboard/demography" },
      { label: "Governance & Institutions", to: "/dashboard/governance" },
      { label: "Technology & Innovation", to: "/dashboard/tech-innovation" },
      { label: "Health & Wellbeing", to: "/dashboard/health" },
      { label: "Education & Human Capital", to: "/dashboard/education" },
      { label: "Environment", to: "/dashboard/environment" },
    ],
  },
  {
    label: "Other data",
    items: [
      { label: "Yahoo Finance", to: "/yahoo" },
      { label: "Crypto", to: "/crypto" },
      { label: "News Explorer", to: "/news" },
    ],
  },
  {
    label: "Regional Statistics",
    items: [
      { label: "United States (FRED)", to: "/regional/fred" },
      { label: "European Union (Eurostat)", to: "/regional/eurostat" },
    ],
  },
  {
    label: "Constructors",
    items: [
      { label: "Custom Plot", to: "/constructors/custom-plot" },
      { label: "Clustering Sandbox", to: "/constructors/clustering" },
    ],
  },
  {
    label: "AI",
    items: [{ label: "AI Analyst", to: "/ai" }],
  },
];
