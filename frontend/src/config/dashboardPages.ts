/**
 * The 10 config-driven dashboard pages. Each maps a route slug + title to the
 * `world_bank_download_config.json` section(s) it renders — the same page→section
 * mapping the Streamlit pages encode. Slugs match the sidebar nav in `nav.ts`.
 */

export interface DashboardPageDef {
  slug: string;
  title: string;
  sectionKeys: string[];
  caption: string;
}

export const DASHBOARD_PAGES: DashboardPageDef[] = [
  {
    slug: "general-economics",
    title: "General Economics Indicators",
    sectionKeys: ["General Economics Indicators"],
    caption:
      "Track core macroeconomic and structural indicators across countries with map, trend, and distribution views.",
  },
  {
    slug: "economy-structure",
    title: "Economy Structure",
    sectionKeys: ["Structure"],
    caption:
      "Explore the structure of national economies: agriculture, manufacturing, and services as a share of GDP.",
  },
  {
    slug: "finance-monetary",
    title: "Finance and Monetary",
    sectionKeys: ["Finance and Monetary", "Fiscal"],
    caption:
      "Monitor monetary, fiscal, and financial indicators to compare policy stance and macro-financial stability across economies.",
  },
  {
    slug: "trade",
    title: "Trade and External Sector",
    sectionKeys: ["Trade and External sector"],
    caption:
      "Analyze external-sector dynamics through trade, openness, and balance signals across countries and over time.",
  },
  {
    slug: "demography",
    title: "Demography",
    sectionKeys: ["Demography"],
    caption:
      "Explore population size, structure, and demographic dynamics to connect labor and social trends with macroeconomic outcomes.",
  },
  {
    slug: "governance",
    title: "Governance and Institutions",
    sectionKeys: ["Governance and Institutions"],
    caption:
      "Assess institutional quality and governance through the World Bank's Worldwide Governance Indicators.",
  },
  {
    slug: "tech-innovation",
    title: "Technology and Innovations",
    sectionKeys: ["Technology and Innovations"],
    caption:
      "Follow innovation capacity, digital adoption, and R&D-related metrics that shape long-term productivity growth.",
  },
  {
    slug: "health",
    title: "Health and Wellbeing",
    sectionKeys: ["Health and wellbeing"],
    caption:
      "Assess health outcomes and wellbeing indicators that influence human capital, resilience, and long-run economic performance.",
  },
  {
    slug: "education",
    title: "Education and Human Capital",
    sectionKeys: ["Education and Human Capital"],
    caption:
      "Analyze schooling, skills, and human capital indicators linked to productivity, inclusion, and labor market quality.",
  },
  {
    slug: "environment",
    title: "Environment and Sustainability",
    sectionKeys: ["Environment and ecology"],
    caption:
      "Track environmental pressures and sustainability signals that interact with growth, risk, and long-term development.",
  },
];
