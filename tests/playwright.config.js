// Playwright configuration for QuantFlow frontend QA
const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./",
  timeout: 60000,
  retries: 1,
  use: {
    baseURL: "http://127.0.0.1:8080",
    headless: true,
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
