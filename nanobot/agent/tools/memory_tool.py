"""Memory search tool using BM25 scoring."""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _split_chunks(text: str, source: str) -> list[dict]:
    """Split text into chunks by ## headings or double newlines."""
    # Split by ## headings first
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split large parts by double newline
        if len(part) > 1000:
            sub_parts = re.split(r"\n\n+", part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    tokens = _tokenize(sp)
                    if tokens:
                        chunks.append({"text": sp, "source": source, "tokens": tokens, "length": len(tokens)})
        else:
            tokens = _tokenize(part)
            if tokens:
                chunks.append({"text": part, "source": source, "tokens": tokens, "length": len(tokens)})
    return chunks


def _bm25_search(query: str, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """BM25 search over text chunks."""
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return []

    n = len(chunks)
    avgdl = sum(c["length"] for c in chunks) / n

    # Document frequency
    df: Counter = Counter()
    for chunk in chunks:
        for token in set(chunk["tokens"]):
            df[token] += 1

    # IDF for query tokens
    idf = {}
    for t in query_tokens:
        if df[t] > 0:
            idf[t] = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)

    # Score each chunk
    for chunk in chunks:
        tf = Counter(chunk["tokens"])
        dl = chunk["length"]
        score = 0.0
        for t in query_tokens:
            if t in idf and t in tf:
                score += idf[t] * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        chunk["score"] = score

    return sorted(chunks, key=lambda c: c["score"], reverse=True)


class MemorySearchTool(Tool):
    """Search through memory files using BM25 text matching."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search through all memory files (MEMORY.md and daily notes) using keyword matching. "
            "Returns the most relevant snippets. Use this to find past context, decisions, or notes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or phrases)",
                },
                "days": {
                    "type": "integer",
                    "description": "Only search daily notes from the last N days (default: 30). MEMORY.md and topic files are always searched.",
                    "minimum": 1,
                    "maximum": 365,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        days = kwargs.get("days", 30)

        if not query.strip():
            return "Error: query cannot be empty."

        chunks = self._load_chunks(days)
        if not chunks:
            return "No memory files found."

        results = _bm25_search(query, chunks)
        # Filter to positive scores, take top 5
        results = [r for r in results if r["score"] > 0][:5]

        if not results:
            return f"No relevant results found for: {query}"

        output_parts = []
        for i, r in enumerate(results, 1):
            snippet = r["text"][:500]
            if len(r["text"]) > 500:
                snippet += "..."
            output_parts.append(f"### Result {i} (score: {r['score']:.2f})\n**File:** {r['source']}\n\n{snippet}")

        return "\n\n---\n\n".join(output_parts)

    def _load_chunks(self, days: int) -> list[dict]:
        """Load and chunk all relevant memory files."""
        from datetime import datetime, timedelta

        chunks = []

        if not self.memory_dir.exists():
            return chunks

        # Always load MEMORY.md
        memory_file = self.memory_dir / "MEMORY.md"
        if memory_file.exists():
            text = memory_file.read_text(encoding="utf-8")
            chunks.extend(_split_chunks(text, "MEMORY.md"))

        # Load topic files (non-date .md files, excluding MEMORY.md)
        for f in self.memory_dir.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            # Skip date-pattern files here, handle them separately with day filter
            if re.match(r"\d{4}-\d{2}-\d{2}\.md$", f.name):
                continue
            text = f.read_text(encoding="utf-8")
            chunks.extend(_split_chunks(text, f.name))

        # Load daily notes within day range
        cutoff = datetime.now().date() - timedelta(days=days)
        for f in self.memory_dir.glob("????-??-??.md"):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_date >= cutoff:
                text = f.read_text(encoding="utf-8")
                chunks.extend(_split_chunks(text, f.name))

        return chunks
