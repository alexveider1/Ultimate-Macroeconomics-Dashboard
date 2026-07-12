import { expect, test } from "@playwright/test";

import { expectNoException, gotoNav, openApp, RENDER_TIMEOUT } from "./helpers";

/**
 * The interactive pages: the AI Analyst chat (multimodal input widgets), the
 * two constructors (custom plot, clustering), and the news explorer. These
 * assert the controls render and the page is driveable — they deliberately do
 * NOT fire live LLM/agent calls (those need real API keys and are out of scope
 * for a frontend smoke).
 */

test.describe("AI Analyst chat", () => {
  test.beforeEach(async ({ page }) => {
    await openApp(page);
    await gotoNav(page, "AI Analyst");
    await expect(page.getByRole("heading", { name: "AI Analyst", level: 1 })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
  });

  test("exposes the multimodal chat + voice inputs", async ({ page }) => {
    await expect(page.getByTestId("stChatInput")).toBeVisible({ timeout: RENDER_TIMEOUT });
    await expect(page.getByTestId("stAudioInput")).toBeVisible();
    await expectNoException(page);
  });

  test("accepts typed input without submitting", async ({ page }) => {
    const box = page.getByTestId("stChatInput").locator("textarea");
    await box.fill("What is GDP per capita for Germany?");
    await expect(box).toHaveValue("What is GDP per capita for Germany?");
    await expectNoException(page);
  });
});

test.describe("constructors", () => {
  test("custom plot builder renders selectors and a chart", async ({ page }) => {
    await openApp(page);
    await gotoNav(page, "Custom Plot Constructor");
    await expect(
      page.getByRole("heading", { name: "Custom Plot Constructor", level: 1 }),
    ).toBeVisible({ timeout: RENDER_TIMEOUT });

    await expect(page.getByText("Select category")).toBeVisible();
    await expect(page.getByText("Select indicator")).toBeVisible();
    await expect(page.getByTestId("stPlotlyChart").first()).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expectNoException(page);
  });

  test("clustering sandbox renders its configuration form", async ({ page }) => {
    await openApp(page);
    await gotoNav(page, "Clustering Sandbox");
    await expect(page.getByRole("heading", { name: "Clustering Sandbox", level: 1 })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });

    // The run trigger is the anchor control; not clicked (needs the clustering
    // service + a full matrix build).
    await expect(page.getByRole("button", { name: "Run clustering" })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expectNoException(page);
  });
});

test.describe("news explorer", () => {
  test("loads without a traceback", async ({ page }) => {
    await openApp(page);
    await gotoNav(page, "News Explorer");
    await expect(page.getByRole("heading", { name: "News Explorer", level: 1 })).toBeVisible({
      timeout: RENDER_TIMEOUT,
    });
    await expectNoException(page);
  });
});
