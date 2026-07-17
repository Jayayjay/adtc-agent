"""
arXiv paper search tool.

CAVEAT: this requires network access to arxiv.org's API, which conflicts
with the offline-first constraint of this submission (Healthcare & Medical /
IMCI decision support -- see src/tools/imci_protocol.py). Not registered by
default in src/tools/registry.py. Kept here in case a future tool needs a
similar network-access pattern as a reference, not because this specific
tool is expected to be used in the final submission.
"""

from __future__ import annotations

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from src.core.exceptions import ToolExecutionError
from src.tools.base import BaseTool

_ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivTool(BaseTool):
    name = "arxiv_search"
    description = "Search arXiv for papers matching a query. Requires network access."
    parameters = {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 3},
    }

    def run(self, query: str, max_results: int = 3) -> dict:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        })
        url = f"{_ARXIV_API}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                raw = resp.read()
        except Exception as e:
            raise ToolExecutionError(
                self.name,
                f"Network request failed ({e}). This tool requires internet access; "
                "if running fully offline, remove it from the registry.",
            )

        root = ET.fromstring(raw)
        results = []
        for entry in root.findall("atom:entry", _NS):
            title = entry.findtext("atom:title", default="", namespaces=_NS).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=_NS).strip()
            link = entry.findtext("atom:id", default="", namespaces=_NS).strip()
            results.append({"title": title, "summary": summary[:300], "link": link})

        return {"results": results}