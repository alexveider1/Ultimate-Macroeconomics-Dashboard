import { test } from "@playwright/test";

import { expectHeading, expectNoException, gotoNav, openApp, PAGES } from "./helpers";

/**
 * Full navigation sweep: open the app once and walk every sidebar page in
 * order, asserting each renders its heading with no Python traceback. This is
 * the primary regression net that every page still loads after a change.
 *
 * Runs serially on a single session (like a real user clicking through the
 * sidebar), so a broken page fails fast and points at the exact route.
 */
test.describe.configure({ mode: "serial" });

test.describe("navigate every page", () => {
  test("walks the whole sidebar", async ({ page }) => {
    test.slow(); // 18 heavy, data-backed pages on one session.
    await openApp(page);

    for (const { nav, heading } of PAGES) {
      await test.step(`open "${nav}"`, async () => {
        await gotoNav(page, nav);
        await expectHeading(page, heading);
        await expectNoException(page);
      });
    }
  });
});
