import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.coverage_index import build_index


class CoverageIndexTests(unittest.TestCase):
    def test_build_index_normalizes_metadata_fields(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            filing_dir = data_root / "filings" / "AAPL" / "10-Q" / "FY2024" / "Q1"
            filing_dir.mkdir(parents=True)
            (filing_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "ticker": " aapl ",
                        "form": " 10-q ",
                        "fiscal_year_label": " FY2024 ",
                        "quarter_label": " Q1 ",
                        "filing_date": " 2024-05-03 ",
                        "accession": " 0000320193-24-000069 ",
                        "filing_chars": "12345",
                        "section_keys": ["item-1-financial-statements"],
                        "exhibits": ["EX-99.1"],
                    }
                ),
                encoding="utf-8",
            )

            universe_path = root / "tech_universe.json"
            universe_path.write_text(
                json.dumps(
                    {
                        "tickers": [
                            {
                                "ticker": " aapl ",
                                "cik": 320193,
                                "company_name": " Apple Inc. ",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            index = build_index(data_root=data_root, universe_path=universe_path)

        self.assertEqual(index["stats"]["by_form"], {"10-Q": 1})
        self.assertEqual(index["companies"][0]["ticker"], "AAPL")
        self.assertEqual(index["companies"][0]["cik"], "320193")
        self.assertEqual(index["companies"][0]["company_name"], "Apple Inc.")

        filing = index["filings"][0]
        self.assertEqual(filing["ticker"], "AAPL")
        self.assertEqual(filing["company_name"], "Apple Inc.")
        self.assertEqual(filing["form"], "10-Q")
        self.assertEqual(filing["fiscal_year_label"], "FY2024")
        self.assertEqual(filing["quarter_label"], "Q1")
        self.assertEqual(filing["period_label"], "FY2024 Q1")
        self.assertEqual(filing["filing_date"], "2024-05-03")
        self.assertEqual(filing["accession"], "0000320193-24-000069")
        self.assertEqual(filing["filing_chars"], 12345)
        self.assertEqual(filing["section_count"], 1)
        self.assertEqual(filing["exhibit_count"], 1)

    def test_build_index_uses_zero_for_invalid_filing_chars(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            filing_dir = data_root / "filings" / "MSFT" / "8-K" / "2024-01-30"
            filing_dir.mkdir(parents=True)
            (filing_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "ticker": "MSFT",
                        "form": "8-K",
                        "filing_date": "2024-01-30",
                        "filing_chars": "not-a-number",
                    }
                ),
                encoding="utf-8",
            )

            universe_path = root / "tech_universe.json"
            universe_path.write_text(json.dumps({"tickers": []}), encoding="utf-8")

            index = build_index(data_root=data_root, universe_path=universe_path)

        self.assertEqual(index["filings"][0]["filing_chars"], 0)
        self.assertEqual(index["filings"][0]["period_label"], "filed 2024-01-30")


if __name__ == "__main__":
    unittest.main()
