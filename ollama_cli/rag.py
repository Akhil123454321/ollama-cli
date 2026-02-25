#!/usr/bin/env python3
"""
rag.py — CodeMinder-style RAG for Ollama CLI
=============================================
AST-aware code chunking + local embeddings + LanceDB vector store.

Install deps:
    pip install lancedb sentence-transformers tree-sitter tree-sitter-python

Features:
  - AST-aware chunking via tree-sitter (functions, classes, top-level blocks)
  - Falls back to line-window chunking for non-Python files
  - Local embeddings via sentence-transformers (all-MiniLM-L6-v2, runs fully offline)
  - LanceDB embedded vector store (no server, single directory)
  - Incremental reconciliation — only re-indexes files that changed (mtime)
  - Auto-indexes on launch when inside a git repo
  - /rag index    — manual full or incremental index
  - /rag search   — semantic search, returns ranked chunks
  - /rag status   — index stats
  - /rag clear    — wipe the index

Architecture:
    RAGIndex          — main public class, owns the DB and embedder
    _Embedder         — wraps SentenceTransformer, lazy-loaded
    _ASTChunker       — tree-sitter based chunker for Python
    _LineChunker      — sliding-window fallback for any text file
    _FileWatcher      — mtime-based change detection
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# Suppress HuggingFace tokenizers parallelism warning when used in threaded context
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Suppress HuggingFace download progress bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Optional heavy deps — all guarded ────────────────────────────────────────

try:
    import lancedb
    import pyarrow as pa
    LANCEDB_OK = True
except ImportError:
    LANCEDB_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    TS_OK = True
except ImportError:
    TS_OK = False


# ── Constants ─────────────────────────────────────────────────────────────────

DB_DIR_NAME      = ".ollama_rag"          # stored inside project root
EMBED_MODEL      = "all-MiniLM-L6-v2"    # 22MB, fast, no trust_remote_code needed
TABLE_NAME       = "chunks"
CHUNK_LINES      = 40                     # fallback line-window size
CHUNK_OVERLAP    = 8                      # overlap for line-window chunks
MAX_CHUNK_CHARS  = 2000                   # hard cap per chunk
MAX_FILE_SIZE    = 512 * 1024             # skip files > 512KB
EMBED_DIM        = 384                           # dimension for all-MiniLM-L6-v2

# Files and dirs to always skip
IGNORE_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build", "*.egg-info",
    ".ollama_rag", ".tox", "htmlcov", ".ruff_cache",
}
IGNORE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".bin", ".exe", ".whl", ".lock",
    ".DS_Store", ".gitignore",
}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".env",
    ".html", ".css", ".scss", ".sql", ".graphql", ".proto",
    ".tf", ".hcl", ".dockerfile", "Dockerfile", ".makefile", "Makefile",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One indexable unit of code/text."""
    file_path:   str          # relative to project root
    start_line:  int          # 1-indexed
    end_line:    int
    content:     str
    chunk_type:  str          # "function" | "class" | "block" | "window"
    symbol_name: str = ""     # function/class name if known
    language:    str = ""     # "python" | "javascript" | etc.
    file_mtime:  float = 0.0

    @property
    def chunk_id(self) -> str:
        h = hashlib.md5(f"{self.file_path}:{self.start_line}:{self.content[:64]}".encode()).hexdigest()[:12]
        return h

    @property
    def display_location(self) -> str:
        loc = f"{self.file_path}:{self.start_line}"
        if self.symbol_name:
            loc += f" ({self.symbol_name})"
        return loc


@dataclass
class SearchResult:
    chunk:    Chunk
    score:    float           # cosine similarity, higher = better
    rank:     int


# ── Embedder ──────────────────────────────────────────────────────────────────

class _Embedder:
    """Lazy-loaded SentenceTransformer wrapper."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _load(self):
        if self._model is None:
            if not ST_OK:
                raise ImportError(
                    "sentence-transformers not installed.\n"
                    "Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vecs.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ── AST Chunker (Python) ──────────────────────────────────────────────────────

class _ASTChunker:
    """
    Tree-sitter based chunker.
    Strategy from CodeMinder paper: extract largest named AST nodes
    (functions, classes) as chunks. For top-level code between named nodes,
    emit as "block" chunks.
    """

    def __init__(self):
        if not TS_OK:
            raise ImportError(
                "tree-sitter or tree-sitter-python not installed.\n"
                "Run: pip install tree-sitter tree-sitter-python"
            )
        PY_LANG = Language(tspython.language())
        self._parser = Parser(PY_LANG)

    def chunk(self, source: str, file_path: str, mtime: float) -> List[Chunk]:
        lines = source.splitlines()
        tree  = self._parser.parse(source.encode())
        chunks: List[Chunk] = []
        covered_lines: set[int] = set()

        def _extract_node(node, node_type: str, name_field: str = "name"):
            """Recursively extract named nodes of the given type."""
            for child in node.children:
                if child.type in ("function_definition", "async_function_definition", "class_definition"):
                    start = child.start_point[0]   # 0-indexed
                    end   = child.end_point[0]
                    # Get symbol name
                    name_node = child.child_by_field_name("name")
                    symbol = name_node.text.decode() if name_node else ""
                    content = "\n".join(lines[start:end + 1])
                    # Cap chunk size
                    if len(content) > MAX_CHUNK_CHARS:
                        content = content[:MAX_CHUNK_CHARS] + "\n# ... (truncated)"
                    chunk_type = "class" if child.type == "class_definition" else "function"
                    chunks.append(Chunk(
                        file_path   = file_path,
                        start_line  = start + 1,
                        end_line    = end + 1,
                        content     = content,
                        chunk_type  = chunk_type,
                        symbol_name = symbol,
                        language    = "python",
                        file_mtime  = mtime,
                    ))
                    for ln in range(start, end + 1):
                        covered_lines.add(ln)
                    # Recurse into class bodies for methods
                    if child.type == "class_definition":
                        body = child.child_by_field_name("body")
                        if body:
                            _extract_node(body, "method")

        _extract_node(tree.root_node, "top")

        # Emit uncovered lines as "block" chunks (imports, top-level statements)
        uncovered = sorted(set(range(len(lines))) - covered_lines)
        if uncovered:
            # Group contiguous uncovered ranges
            groups: List[List[int]] = []
            current: List[int] = [uncovered[0]]
            for ln in uncovered[1:]:
                if ln == current[-1] + 1:
                    current.append(ln)
                else:
                    groups.append(current)
                    current = [ln]
            groups.append(current)

            for group in groups:
                content = "\n".join(lines[ln] for ln in group).strip()
                if not content or len(content) < 10:
                    continue
                if len(content) > MAX_CHUNK_CHARS:
                    content = content[:MAX_CHUNK_CHARS] + "\n# ... (truncated)"
                chunks.append(Chunk(
                    file_path   = file_path,
                    start_line  = group[0] + 1,
                    end_line    = group[-1] + 1,
                    content     = content,
                    chunk_type  = "block",
                    language    = "python",
                    file_mtime  = mtime,
                ))

        return chunks


# ── Line-window Chunker (fallback) ────────────────────────────────────────────

class _LineChunker:
    """
    Sliding-window line chunker for non-Python files.
    Uses CHUNK_LINES lines per window with CHUNK_OVERLAP lines of overlap.
    """

    def chunk(self, source: str, file_path: str, mtime: float, language: str = "") -> List[Chunk]:
        lines   = source.splitlines()
        chunks: List[Chunk] = []
        step    = max(1, CHUNK_LINES - CHUNK_OVERLAP)
        i       = 0
        while i < len(lines):
            window = lines[i : i + CHUNK_LINES]
            content = "\n".join(window).strip()
            if content and len(content) >= 10:
                if len(content) > MAX_CHUNK_CHARS:
                    content = content[:MAX_CHUNK_CHARS] + "\n... (truncated)"
                chunks.append(Chunk(
                    file_path  = file_path,
                    start_line = i + 1,
                    end_line   = min(i + CHUNK_LINES, len(lines)),
                    content    = content,
                    chunk_type = "window",
                    language   = language,
                    file_mtime = mtime,
                ))
            i += step
        return chunks


# ── File Watcher ──────────────────────────────────────────────────────────────

class _FileWatcher:
    """
    Tracks file mtimes. Persisted as a JSON sidecar in the DB dir.
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._state: dict[str, float] = {}
        self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text())
            except Exception:
                self._state = {}

    def _save(self):
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def has_changed(self, path: Path, rel_path: str) -> bool:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        return self._state.get(rel_path, 0.0) != mtime

    def mark_indexed(self, rel_path: str, mtime: float):
        self._state[rel_path] = mtime
        self._save()

    def remove(self, rel_path: str):
        self._state.pop(rel_path, None)
        self._save()

    def all_indexed(self) -> set[str]:
        return set(self._state.keys())


# ── RAGIndex — main public class ──────────────────────────────────────────────

class RAGIndex:
    """
    Public interface. One instance per project root.

    Usage:
        rag = RAGIndex(project_root)
        rag.index()                         # index/reconcile
        results = rag.search("auth logic")
        rag.status()                        # returns dict
    """

    def __init__(self, project_root: Path):
        if not LANCEDB_OK:
            raise ImportError(
                "lancedb or pyarrow not installed.\n"
                "Run: pip install lancedb"
            )
        self.root     = Path(project_root)
        self.db_dir   = self.root / DB_DIR_NAME
        self.db_dir.mkdir(exist_ok=True)

        self._embedder    = _Embedder()
        self._ast_chunker: Optional[_ASTChunker] = None
        self._line_chunker = _LineChunker()
        self._watcher     = _FileWatcher(self.db_dir / "mtime_state.json")

        # LanceDB setup
        self._db    = lancedb.connect(str(self.db_dir / "vectors"))
        self._table = self._get_or_create_table()

    # ── Table management ──────────────────────────────────────────────────────

    def _schema(self):
        return pa.schema([
            pa.field("chunk_id",    pa.string()),
            pa.field("file_path",   pa.string()),
            pa.field("start_line",  pa.int32()),
            pa.field("end_line",    pa.int32()),
            pa.field("content",     pa.string()),
            pa.field("chunk_type",  pa.string()),
            pa.field("symbol_name", pa.string()),
            pa.field("language",    pa.string()),
            pa.field("file_mtime",  pa.float64()),
            pa.field("vector",      pa.list_(pa.float32(), EMBED_DIM)),
        ])

    def _get_or_create_table(self):
        tables = self._db.table_names()
        if TABLE_NAME in tables:
            return self._db.open_table(TABLE_NAME)
        return self._db.create_table(TABLE_NAME, schema=self._schema())

    # ── Chunking helpers ──────────────────────────────────────────────────────

    def _get_ast_chunker(self) -> Optional[_ASTChunker]:
        if self._ast_chunker is not None:
            return self._ast_chunker
        try:
            self._ast_chunker = _ASTChunker()
            return self._ast_chunker
        except ImportError:
            return None

    def _chunk_file(self, path: Path, rel_path: str, mtime: float) -> List[Chunk]:
        try:
            source = path.read_text(errors="replace")
        except Exception:
            return []

        ext = path.suffix.lower()
        lang = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript", ".rs": "rust",
            ".go": "go", ".java": "java", ".rb": "ruby", ".cpp": "cpp",
            ".c": "c", ".cs": "csharp", ".sh": "bash", ".md": "markdown",
        }.get(ext, ext.lstrip("."))

        if ext == ".py":
            chunker = self._get_ast_chunker()
            if chunker:
                try:
                    return chunker.chunk(source, rel_path, mtime)
                except Exception:
                    pass  # fall through to line chunker

        return self._line_chunker.chunk(source, rel_path, mtime, language=lang)

    # ── File discovery ────────────────────────────────────────────────────────

    def _iter_files(self) -> Iterator[Tuple[Path, str]]:
        """Yield (absolute_path, relative_path) for all indexable files."""
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            # Check ignored dirs
            parts = path.relative_to(self.root).parts
            if any(part in IGNORE_DIRS or part.endswith(".egg-info") for part in parts):
                continue
            # Check extension
            ext = path.suffix.lower()
            name = path.name
            if ext in IGNORE_EXTS:
                continue
            if ext not in TEXT_EXTS and name not in TEXT_EXTS:
                continue
            # Check file size
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            rel = str(path.relative_to(self.root))
            yield path, rel

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, full: bool = False, progress_cb=None) -> dict:
        """
        Index or reconcile the project.

        Args:
            full:        If True, wipe and reindex everything.
            progress_cb: Optional callable(message: str) for progress updates.

        Returns:
            dict with keys: indexed, skipped, deleted, errors, elapsed
        """
        t0 = time.time()

        def log(msg):
            if progress_cb:
                progress_cb(msg)

        if full:
            log("Clearing existing index...")
            self._db.drop_table(TABLE_NAME)
            self._table = self._get_or_create_table()
            self._watcher._state = {}
            self._watcher._save()

        indexed = 0
        skipped = 0
        errors  = 0

        # Collect files to index
        to_index: List[Tuple[Path, str, float]] = []
        current_files: set[str] = set()

        for path, rel in self._iter_files():
            current_files.add(rel)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if not full and not self._watcher.has_changed(path, rel):
                skipped += 1
                continue
            to_index.append((path, rel, mtime))

        # Find deleted files
        indexed_files = self._watcher.all_indexed()
        deleted_files = indexed_files - current_files
        deleted = len(deleted_files)
        for rel in deleted_files:
            self._remove_file_chunks(rel)
            self._watcher.remove(rel)

        if deleted:
            log(f"Removed {deleted} deleted file(s) from index")

        total = len(to_index)
        if total == 0:
            log(f"Index up to date — {skipped} files unchanged")
            return {"indexed": 0, "skipped": skipped, "deleted": deleted, "errors": 0,
                    "elapsed": time.time() - t0}

        log(f"Indexing {total} file(s)...")

        # Batch embed for efficiency
        BATCH = 32
        batch_chunks: List[Chunk] = []
        file_count = 0

        def _flush_batch():
            nonlocal indexed
            if not batch_chunks:
                return
            texts = [c.content for c in batch_chunks]
            try:
                vectors = self._embedder.embed(texts)
            except Exception as e:
                nonlocal errors
                errors += len(batch_chunks)
                log(f"Embedding error: {e}")
                batch_chunks.clear()
                return

            rows = []
            for chunk, vec in zip(batch_chunks, vectors):
                rows.append({
                    "chunk_id":    chunk.chunk_id,
                    "file_path":   chunk.file_path,
                    "start_line":  chunk.start_line,
                    "end_line":    chunk.end_line,
                    "content":     chunk.content,
                    "chunk_type":  chunk.chunk_type,
                    "symbol_name": chunk.symbol_name,
                    "language":    chunk.language,
                    "file_mtime":  chunk.file_mtime,
                    "vector":      [float(x) for x in vec],
                })
            try:
                self._table.add(rows)
                indexed += len(rows)
            except Exception as e:
                errors += len(rows)
                log(f"DB write error: {e}")
            batch_chunks.clear()

        for i, (path, rel, mtime) in enumerate(to_index):
            if progress_cb and (i % 10 == 0 or i == total - 1):
                log(f"  [{i+1}/{total}] {rel}")

            # Remove old chunks for this file before re-adding
            self._remove_file_chunks(rel)

            chunks = self._chunk_file(path, rel, mtime)
            if not chunks:
                self._watcher.mark_indexed(rel, mtime)
                continue

            batch_chunks.extend(chunks)
            self._watcher.mark_indexed(rel, mtime)
            file_count += 1

            if len(batch_chunks) >= BATCH:
                _flush_batch()

        _flush_batch()  # remaining

        elapsed = time.time() - t0
        log(f"Done — {indexed} chunks from {file_count} files in {elapsed:.1f}s")

        return {
            "indexed": indexed, "skipped": skipped,
            "deleted": deleted, "errors":  errors,
            "elapsed": elapsed,
        }

    def _remove_file_chunks(self, rel_path: str):
        """Delete all chunks belonging to a file."""
        try:
            self._table.delete(f"file_path = '{rel_path}'")
        except Exception:
            pass

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8, file_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Semantic search over the index.

        Args:
            query:       Natural language or code query.
            top_k:       Number of results to return.
            file_filter: Optional glob-style substring to restrict to certain files.

        Returns:
            List of SearchResult sorted by score descending.
        """
        try:
            q_vec = self._embedder.embed_one(query)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

        try:
            q = self._table.search(q_vec).limit(top_k * 2)  # fetch more, then filter
            if file_filter:
                q = q.where(f"file_path LIKE '%{file_filter}%'")
            rows = q.to_list()
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

        results: List[SearchResult] = []
        seen_ids: set[str] = set()

        for row in rows:
            cid = row.get("chunk_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            chunk = Chunk(
                file_path   = row["file_path"],
                start_line  = row["start_line"],
                end_line    = row["end_line"],
                content     = row["content"],
                chunk_type  = row["chunk_type"],
                symbol_name = row.get("symbol_name", ""),
                language    = row.get("language", ""),
                file_mtime  = row.get("file_mtime", 0.0),
            )
            # LanceDB returns _distance (L2) — convert to similarity score
            dist  = row.get("_distance", 0.0)
            score = max(0.0, 1.0 - dist / 2.0)   # approximate cosine from normalized vecs

            results.append(SearchResult(chunk=chunk, score=score, rank=0))
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def search_as_context(self, query: str, top_k: int = 5) -> str:
        """
        Search and format results as a context string for injection into the LLM prompt.
        Called automatically from _build_system() when a query is detected.
        """
        try:
            results = self.search(query, top_k=top_k)
        except Exception:
            return ""

        if not results:
            return ""

        parts = [f"[Relevant code from project — retrieved by semantic search for: '{query}']"]
        for r in results:
            header = f"### {r.chunk.display_location}  (score: {r.chunk.chunk_type})"
            parts.append(f"{header}\n```{r.chunk.language}\n{r.chunk.content}\n```")

        return "\n\n".join(parts)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return index statistics."""
        try:
            total_chunks = self._table.count_rows()
        except Exception:
            total_chunks = 0

        indexed_files = len(self._watcher.all_indexed())

        # Count chunks by language
        by_lang: dict[str, int] = {}
        try:
            rows = self._table.to_pandas()[["language"]].value_counts().to_dict()
            for (lang,), count in rows.items():
                by_lang[lang] = count
        except Exception:
            pass

        return {
            "total_chunks":  total_chunks,
            "indexed_files": indexed_files,
            "by_language":   by_lang,
            "db_path":       str(self.db_dir),
            "embed_model":   EMBED_MODEL,
            "ast_chunking":  TS_OK,
        }

    def clear(self):
        """Wipe the entire index."""
        self._db.drop_table(TABLE_NAME)
        self._table = self._get_or_create_table()
        self._watcher._state = {}
        self._watcher._save()


# ── Dependency check helper ───────────────────────────────────────────────────

def check_deps() -> Tuple[bool, List[str]]:
    """
    Returns (all_ok, list_of_missing_install_commands).
    Call this before constructing RAGIndex to give user a helpful error.
    """
    missing = []
    if not LANCEDB_OK:
        missing.append("pip install lancedb")
    if not ST_OK:
        missing.append("pip install sentence-transformers")
    if not TS_OK:
        missing.append("pip install tree-sitter tree-sitter-python")
    return len(missing) == 0, missing


# ── CLI integration helpers ───────────────────────────────────────────────────

def format_search_results_rich(results: List[SearchResult]) -> str:
    """
    Format search results as Rich-markup string for console.print().
    Used by cmd_rag in main.py.
    """
    if not results:
        return "[dim]No results found.[/dim]"

    lines = []
    for r in results:
        score_bar = "█" * int(r.score * 10) + "░" * (10 - int(r.score * 10))
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.4 else "red"
        lines.append(
            f"  [{score_color}]{r.rank}.[/{score_color}] "
            f"[cyan]{r.chunk.file_path}[/cyan]"
            f"[dim]:{r.chunk.start_line}[/dim]"
            + (f"  [bold]{r.chunk.symbol_name}[/bold]" if r.chunk.symbol_name else "")
            + f"  [{score_color}]{score_bar}[/{score_color}] [dim]{r.score:.2f}[/dim]"
        )
        # Preview — first 2 non-empty lines
        preview_lines = [l.strip() for l in r.chunk.content.splitlines() if l.strip()][:2]
        for pl in preview_lines:
            lines.append(f"     [dim]{pl[:100]}[/dim]")
        lines.append("")

    return "\n".join(lines)
