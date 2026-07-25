from __future__ import annotations

import os
import unittest

from quantflow.data.finviz_client import _apply_proxy_env_from_qf, get_scraping_runtime_config


class FinvizProxyEnvTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (
            "QF_SCRAPING_PROXY_URL",
            "SENSITIVE_LOOKUP_PROXY_URL",
            "SMARTPROXY_PROXY_URL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "QF_SCRAPING_PROVIDER",
            "SCRAPING_API_KEY",
            "SCRAPINGBEE_API_KEY",
            "SCRAPINGBEE_PROXY_URL",
            "SCRAPE_DO_PROXY_URL",
            "SCRAPE_DO_ENDPOINT",
            "SCRAPINGBEE_ENDPOINT",
        )}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in self._saved:
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_uses_primary_qf_proxy_variable(self):
        os.environ["QF_SCRAPING_PROXY_URL"] = "http://proxy-a:3128"
        _apply_proxy_env_from_qf()
        self.assertEqual(os.environ.get("HTTP_PROXY"), "http://proxy-a:3128")
        self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://proxy-a:3128")

    def test_uses_sensitive_lookup_alias(self):
        os.environ["SENSITIVE_LOOKUP_PROXY_URL"] = "http://proxy-b:3128"
        _apply_proxy_env_from_qf()
        self.assertEqual(os.environ.get("HTTP_PROXY"), "http://proxy-b:3128")
        self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://proxy-b:3128")

    def test_runtime_config_detects_provider_and_proxy_source(self):
        os.environ["QF_SCRAPING_PROVIDER"] = "scrapingbee"
        os.environ["SCRAPINGBEE_PROXY_URL"] = "http://proxy-c:8080"
        cfg = get_scraping_runtime_config()
        self.assertEqual(cfg.provider, "scrapingbee")
        self.assertTrue(cfg.proxy_configured)
        self.assertEqual(cfg.proxy_env_source, "SCRAPINGBEE_PROXY_URL")
        self.assertEqual(cfg.mode, "proxy")

    def test_runtime_config_warns_when_provider_without_proxy(self):
        os.environ["QF_SCRAPING_PROVIDER"] = "scrape_do"
        os.environ.pop("SCRAPING_API_KEY", None)
        cfg = get_scraping_runtime_config()
        self.assertEqual(cfg.provider, "scrape_do")
        self.assertFalse(cfg.proxy_configured)
        self.assertTrue(cfg.notes)

    def test_runtime_config_uses_api_mode_for_scrape_do_key(self):
        os.environ["QF_SCRAPING_PROVIDER"] = "scrape_do"
        os.environ["SCRAPING_API_KEY"] = "x"
        os.environ["SCRAPE_DO_ENDPOINT"] = "https://api.scrape.do"
        cfg = get_scraping_runtime_config()
        self.assertEqual(cfg.mode, "api")
        self.assertEqual(cfg.api_endpoint, "https://api.scrape.do")

    def test_runtime_config_uses_api_mode_for_scrapingbee_key(self):
        os.environ["QF_SCRAPING_PROVIDER"] = "scrapingbee"
        os.environ["SCRAPINGBEE_API_KEY"] = "x"
        os.environ["SCRAPINGBEE_ENDPOINT"] = "https://app.scrapingbee.com/api/v1/"
        cfg = get_scraping_runtime_config()
        self.assertEqual(cfg.mode, "api")
        self.assertEqual(cfg.api_endpoint, "https://app.scrapingbee.com/api/v1/")


if __name__ == "__main__":
    unittest.main()
