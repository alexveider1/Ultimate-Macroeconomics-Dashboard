import { expect, test } from "@playwright/test";

/**
 * Placeholder smoke test. Skipped until the new frontend lands and the stack
 * is running locally — remove `.skip` (or run with the app up) to exercise it.
 */
test.skip("dashboard loads", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle(/.+/);
});
