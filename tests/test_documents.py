import unittest
from io import BytesIO

from agentx.documents import DocumentExtractor, DocumentExtractionError


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, stream: BytesIO):
        self.stream = stream
        self.pages = [_FakePage("first page"), _FakePage("second page")]


class DocumentExtractorTests(unittest.TestCase):
    def test_html_extraction_suppresses_scripts_and_bounds_text(self):
        extractor = DocumentExtractor(max_chars=20)

        result = extractor.extract(
            b"<html><title> Guide </title><script>secret()</script><body>Useful text here</body></html>",
            media_type="text/html",
        )

        self.assertEqual("Guide", result.title)
        self.assertEqual("Useful text here", result.text)
        self.assertNotIn("secret", result.text)

    def test_json_is_pretty_printed_and_bounded(self):
        result = DocumentExtractor(max_chars=100).extract(b'{"name":"AgentX","ok":true}', media_type="application/json")

        self.assertIn('"name": "AgentX"', result.text)
        self.assertFalse(result.truncated)

    def test_pdf_extraction_is_bounded_and_reports_page_count(self):
        result = DocumentExtractor(max_chars=20, max_pages=1, pdf_reader_factory=_FakeReader).extract(
            b"not a real pdf", media_type="application/pdf"
        )

        self.assertEqual("first page", result.text)
        self.assertEqual(2, result.page_count)
        self.assertTrue(result.truncated)

    def test_unsupported_type_is_rejected(self):
        with self.assertRaisesRegex(DocumentExtractionError, "unsupported"):
            DocumentExtractor().extract(b"binary", media_type="application/octet-stream")


if __name__ == "__main__":
    unittest.main()
