import { createBrowserRouter } from "react-router-dom";

import { IndicatorPage } from "@/components/IndicatorPage";
import { AppLayout } from "@/components/layout/AppLayout";
import { DASHBOARD_PAGES } from "@/config/dashboardPages";
import { AiChatPage } from "@/pages/AiChatPage";
import { ClusteringSandboxPage } from "@/pages/ClusteringSandboxPage";
import { CryptoPage } from "@/pages/CryptoPage";
import { CustomPlotPage } from "@/pages/CustomPlotPage";
import { EurostatRegionalPage } from "@/pages/EurostatRegionalPage";
import { FredRegionalPage } from "@/pages/FredRegionalPage";
import { NewsPage } from "@/pages/NewsPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { YahooPage } from "@/pages/YahooPage";

const dashboardRoutes = DASHBOARD_PAGES.map((page) => ({
  path: `dashboard/${page.slug}`,
  element: (
    <IndicatorPage title={page.title} sectionKeys={page.sectionKeys} caption={page.caption} />
  ),
}));

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <OverviewPage /> },
      ...dashboardRoutes,
      { path: "constructors/custom-plot", element: <CustomPlotPage /> },
      { path: "yahoo", element: <YahooPage /> },
      { path: "crypto", element: <CryptoPage /> },
      { path: "news", element: <NewsPage /> },
      { path: "regional/fred", element: <FredRegionalPage /> },
      { path: "regional/eurostat", element: <EurostatRegionalPage /> },
      { path: "ai", element: <AiChatPage /> },
      { path: "constructors/clustering", element: <ClusteringSandboxPage /> },
    ],
  },
]);
