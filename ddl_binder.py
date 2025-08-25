#!/usr/bin/env python3
"""
DDL Evidence Binder (drop‑in replacement)

Highlights
- Robust parquet discovery (recursive), schema normalization, and diagnostics
- Heartbeat/liveness file to distinguish "thinking" vs. "hung"
- Three-tier retrieval (A strict, B relaxed, C semantic-lite) with tier stamps
- Budgeted NLI labeling with short timeouts + lexical fallback
- Reliable paper anchor extraction (exact + fuzzy) with precise citation spans
- Deterministic lexical retrieval scoring + scan cap for speed
- Bound/paper-only/unbound compatible with your three-tier composer
"""

import os
import re
import json
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

import pandas as pd

# Configure structured logging
logger = logging.getLogger(__name__)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with DDL-specific fields."""
    
    def __init__(self, run_id="unknown", paper_id="unknown"):
        super().__init__()
        self.run_id = run_id
        self.paper_id = paper_id
    
    def format(self, record):
        import json
        from datetime import datetime
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, 'run_id', self.run_id),
            "paper_id": getattr(record, 'paper_id', self.paper_id),
            "message": record.getMessage()
        }
        
        # Add DDL-specific fields if present
        ddl_fields = ['stage', 'event', 'pairs_processed', 'nli_timeouts', 'parquet_files_loaded', 
                     'files_processed', 'rows_processed', 'candidates_found', 'progress_pct']
        
        for field in ddl_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
                
        return json.dumps(log_entry)

# Set default logging level - will be configured per-instance
logger.setLevel(logging.INFO)

# Parquet engine validation with actionable error messages
try:
    import pyarrow as _parquet_engine
    parquet_engine = "pyarrow"
    logger.info(f"Using pyarrow {_parquet_engine.__version__} parquet engine")
except ImportError:
    try:
        import fastparquet as _parquet_engine
        parquet_engine = "fastparquet"
        logger.info(f"Using fastparquet {_parquet_engine.__version__} parquet engine")
    except ImportError:
        print("\n❌ ERROR: No parquet engine found!")
        print("   DDL requires a parquet library to read corpus files.")
        print("   Install one of the following:")
        print("   → pip install pyarrow>=15.0.0  (recommended)")
        print("   → pip install fastparquet>=2024.5.0  (alternative)")
        print("\n   Run preflight checks: python scripts/preflight.py")
        raise RuntimeError("Missing parquet engine - see installation instructions above")

try:
    # Semantic retrieval dependencies
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi
    _HAVE_ST = True
except Exception as e:
    logger.warning(f"Semantic dependencies not available: {e}")
    _HAVE_ST = False

# Resource hygiene (prevents HF fork/tokenizer issues)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import openai  # LM Studio/OpenAI-compatible client


class DDLEvidenceBinder:
    """
    Binds accepted daydreams to supporting quotes from both paper and corpus.
    Two-hop retrieval + NLI labeling + robust fallbacks.
    """

    # ----------- init & corpus loading -----------

    def __init__(self,
                 corpus_path: str,
                 api_base: str = "http://localhost:11434/v1",
                 api_key: str = "ollama",
                 model: str = "qwen3-30b",
                 min_quote_len: int = 50,
                 max_quote_len: int = 120,
                 quotes_per_daydream: int = 3,    # 1 paper + up to 2 corpus
                 max_quotes_total: int = 30,      # global quote budget (used upstream)
                 nli_timeout_s: int = 4,
                 scan_cap: int = 50000,
                 config: Dict[str, Any] = None,
                 run_id: str = None,
                 run_dir: str = None):

        self.corpus_path = os.path.abspath(corpus_path)
        self.client = openai.OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        
        # Configuration and traceability
        self.config = config or {}
        self.run_id = run_id or "unknown"
        self.paper_id = "unknown"  # Will be set when paper is processed
        self.run_dir = run_dir or os.getcwd()  # Default to CWD if not provided
        self.semantic_cfg = self.config.get("binder", {}).get("semantic", {})
        self.parq_cfg = self.config.get("binder", {}).get("parquet", {})
        
        # Ensure run directory exists
        if self.run_dir != os.getcwd():
            os.makedirs(self.run_dir, exist_ok=True)
        
        # Configure structured logging for this instance
        self._setup_logging()

        # Quote constraints
        self.min_quote_length = min_quote_len
        self.max_quote_length = max_quote_len

        # Evidence requirements
        self.quotes_per_daydream = quotes_per_daydream
        self.max_quotes_total = max_quotes_total

        # NLI/throughput knobs
        self.nli_timeout_s = nli_timeout_s
        self.scan_cap = scan_cap  # max rows to scan for lexical retrieval

        # Liveness/telemetry
        self.heartbeat_file = os.path.join(self.run_dir, "binder.heartbeat.json")
        self.last_heartbeat = 0.0

        # Initialize semantic retrieval if enabled
        self.embedding_model = None
        self.faiss_index = None
        self.faiss_ids = None
        
        if self.semantic_cfg.get("enabled", True) and _HAVE_ST:
            try:
                model_name = self.semantic_cfg.get("model_name", "BAAI/bge-m3")
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_model.max_seq_length = 512
                logger.info(f"Semantic retrieval enabled with {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Falling back to lexical-only retrieval")
        else:
            logger.info("Semantic retrieval disabled (config or dependencies)")

        # Initialize metrics tracking
        self.metrics = {
            "run_id": self.run_id,
            "parquet_files_loaded": 0,
            "rows_scanned": 0,
            "unique_docs_scanned": 0,
            "bm25_candidates_considered": 0,
            "semantic_enabled": self.embedding_model is not None,
            "semantic_topk": self.semantic_cfg.get("topk", 200) if self.embedding_model else 0,
            "nli_pairs_evaluated": 0,
            "nli_timeouts": 0,
            "bound_quotes": 0
        }

        # Validate parquet engine before corpus loading
        self._validate_parquet_engine()

        # Load/normalize corpus
        self.files_loaded = 0
        self.corpus_df = self._load_corpus()
        
        # Update metrics after corpus load
        self.metrics["parquet_files_loaded"] = self.files_loaded
        self.metrics["rows_scanned"] = len(self.corpus_df) if not self.corpus_df.empty else 0
        self.metrics["unique_docs_scanned"] = self.corpus_df['doc_id'].nunique() if not self.corpus_df.empty else 0
        
        # Build semantic index if enabled
        if self.embedding_model is not None:
            self._build_semantic_index()

    # ----------- filesystem & schema -----------

    def _discover_parquets(self, corpus_path: str) -> List[str]:
        """Discover parquet files recursively or accept a single parquet file."""
        import glob

        if os.path.isdir(corpus_path):
            files = glob.glob(os.path.join(corpus_path, "**", "*.parquet"), recursive=True)
        elif corpus_path.endswith(".parquet"):
            files = [corpus_path]
        else:
            # Try non-recursive same-dir fallback
            corpus_dir = os.path.dirname(corpus_path) if not os.path.isdir(corpus_path) else corpus_path
            files = glob.glob(os.path.join(corpus_dir, "*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No parquet files found.\n"
                f"  Path: {corpus_path}\n"
                f"  CWD: {os.getcwd()}\n"
                f"  Absolute: {os.path.abspath(corpus_path)}"
            )
        files = sorted(files)
        print(f"✓ Discovered {len(files)} parquet files")
        self._log("info", f"Discovered {len(files)} parquet files", 
                 stage="corpus_discovery", event="files_discovered", files_found=len(files))
        return files

    def _normalize_df(self, df: pd.DataFrame, file_path: str = "unknown") -> pd.DataFrame:
        """Normalize to a consistent schema: doc_id, author, year, title, text."""
        COLUMN_MAPPING = {
            "content": "text",
            "body": "text",
            "chunk": "text",
            "span_text": "text",
            "authors": "author",
            "doc_title": "title",
            "source_title": "title",
        }
        for old, new in COLUMN_MAPPING.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        # doc_id
        if "doc_id" not in df.columns:
            if "source_id" in df.columns:
                df["doc_id"] = df["source_id"].astype(str)
            else:
                df["doc_id"] = "unknown_" + df.index.astype(str)

        # author
        if "author" not in df.columns:
            df["author"] = "Unknown"

        # year
        if "year" not in df.columns:
            if "pub_date" in df.columns:
                try:
                    df["year"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.year.fillna(2020).astype(int)
                except Exception:
                    df["year"] = 2020
            else:
                df["year"] = 2020

        # title
        if "title" not in df.columns:
            df["title"] = "Untitled"

        # text
        if "text" not in df.columns:
            print(f"Warning: No text-like column in {file_path}, creating empty 'text'")
            df["text"] = ""

        # Ensure string types & no NaNs
        for col in ["doc_id", "author", "title", "text"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")

        # Sample schema audit
        if len(df) >= 5:
            sample = df[["doc_id", "author", "year", "title", "text"]].head(5)
            print(f"✓ Schema sample from {file_path}:")
            for idx, row in sample.iterrows():
                preview = row["text"][:60].replace("\n", " ")
                if len(row["text"]) > 60:
                    preview += "..."
                print(f"  [{idx}] {row['doc_id']} | {row['author']} | {row['year']} | {row['title'][:32]}... | '{preview}'")

        # Empty text ratio
        empty_ratio = (df["text"].str.len() == 0).mean() if len(df) else 1.0
        if empty_ratio > 0.30:
            msg = f"⚠️  High empty-text ratio {empty_ratio:.1%} in {file_path}"
            print(msg)
            try:
                with open("binder.skipped.txt", "a", encoding="utf-8") as f:
                    f.write(f"{file_path}: empty_text_ratio={empty_ratio:.1%}\n")
            except Exception:
                pass
        return df

    def _validate_parquet_engine(self):
        """Validate that parquet engine works correctly with actual file operation."""
        try:
            # Test basic parquet functionality with a tiny synthetic dataset
            test_data = pd.DataFrame({
                'doc_id': ['test_doc'],
                'text': ['test content'],
                'author': ['test_author'],
                'year': [2024]
            })
            
            # Try to write and read back a small parquet file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                try:
                    # Write test parquet
                    test_data.to_parquet(tmp.name, engine=parquet_engine)
                    
                    # Read it back
                    read_back = pd.read_parquet(tmp.name, engine=parquet_engine)
                    
                    # Verify basic functionality
                    if len(read_back) != 1 or 'doc_id' not in read_back.columns:
                        raise ValueError("Parquet round-trip test failed")
                        
                    logger.info(f"Parquet engine {parquet_engine} validated successfully")
                    
                finally:
                    # Clean up test file
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
                        
        except Exception as e:
            print(f"\n❌ ERROR: Parquet engine validation failed!")
            print(f"   Engine: {parquet_engine}")
            print(f"   Error: {e}")
            print("   This suggests the parquet library installation is corrupted.")
            print("   Try reinstalling:")
            print(f"   → pip uninstall {parquet_engine}")
            print(f"   → pip install {parquet_engine}>={'15.0.0' if parquet_engine == 'pyarrow' else '2024.5.0'}")
            print("\n   Run preflight checks: python scripts/preflight.py")
            raise RuntimeError(f"Parquet engine {parquet_engine} validation failed: {e}")

    def _setup_logging(self):
        """Configure structured logging for this binder instance."""
        # Create handlers with JSON formatter
        if not hasattr(self, '_log_handler'):
            # Console handler (existing behavior)
            self._log_handler = logging.StreamHandler()
            self._log_formatter = StructuredFormatter(run_id=self.run_id, paper_id=self.paper_id)
            self._log_handler.setFormatter(self._log_formatter)
            
            # File handler for run-scoped logging
            log_file_path = os.path.join(self.run_dir, "binder.log")
            self._file_handler = logging.FileHandler(log_file_path)
            self._file_handler.setFormatter(self._log_formatter)
            
            # Configure logger for this instance
            instance_logger = logging.getLogger(f"{__name__}.{self.run_id}")
            instance_logger.addHandler(self._log_handler)  # Console
            instance_logger.addHandler(self._file_handler)  # File
            instance_logger.setLevel(logging.INFO)
            instance_logger.propagate = False  # Prevent duplicate logs
            
            self.logger = instance_logger

    def _log(self, level, message, **kwargs):
        """Structured logging helper with DDL-specific fields."""
        if not hasattr(self, 'logger'):
            # Fallback to module logger if instance logger not ready
            log_func = getattr(logger, level.lower(), logger.info)
            log_func(message)
            return
            
        # Create log record with extra fields
        extra_fields = {
            'run_id': self.run_id,
            'paper_id': self.paper_id,
        }
        extra_fields.update(kwargs)
        
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message, extra=extra_fields)

    def _load_corpus(self) -> pd.DataFrame:
        """Load and normalize corpus parquets (capped for memory)."""
        try:
            if self.corpus_path.endswith(".parquet"):
                try:
                    df = pd.read_parquet(self.corpus_path)
                    df = self._normalize_df(df, self.corpus_path)
                    self.files_loaded = 1
                    print(f"Loaded corpus with {len(df)} chunks after normalization")
                    self._log("info", f"Single corpus file loaded successfully", 
                             stage="corpus_loading", event="single_file_loaded", 
                             rows_loaded=len(df), file_path=self.corpus_path)
                    return df
                except Exception as e:
                    print(f"\n❌ ERROR: Failed to read corpus parquet file!")
                    print(f"   File: {self.corpus_path}")
                    print(f"   Error: {e}")
                    if "parquet" in str(e).lower() or "arrow" in str(e).lower():
                        print("   This may be a parquet engine issue.")
                        print("   Try: python scripts/preflight.py")
                    print("   Cannot continue without valid corpus data.")
                    raise RuntimeError(f"Failed to load corpus parquet file: {e}")

            files = self._discover_parquets(self.corpus_path)
            dfs, loaded = [], 0
            # Load files using configurable cap and sampling strategy
            default_max_files = self.parq_cfg.get("max_files", 200)
            max_files = min(default_max_files, len(files))
            selected_files = self._sample_parquet_files(files, max_files, "strided")
            print(f"Loading {max_files} of {len(files)} parquet files (strided sampling)...")
            self._log("info", f"Starting multi-file corpus loading", 
                     stage="corpus_loading", event="multi_file_start", 
                     total_files=len(files), selected_files=max_files, sampling_strategy="strided")
            for i, fp in enumerate(selected_files):
                try:
                    chunk = pd.read_parquet(fp)
                    chunk = self._normalize_df(chunk, fp)
                    dfs.append(chunk)
                    loaded += 1
                    
                    # Progress heartbeat every 30 seconds
                    self._progress_heartbeat("corpus_loading", i+1, max_files, 
                                           {"files_loaded": loaded, "current_file": os.path.basename(fp)})
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Loaded {i+1}/{max_files} files...")
                        self._log("info", f"Corpus loading progress", 
                                 stage="corpus_loading", event="progress_update", 
                                 files_processed=i+1, total_files=max_files, 
                                 progress_pct=round((i+1)/max_files*100, 1))
                except Exception as e:
                    print(f"  ❌ Failed to load {fp}: {e}")
                    if "parquet" in str(e).lower() or "arrow" in str(e).lower():
                        print(f"     This may be a parquet engine issue. Try: python scripts/preflight.py")
                    # Continue with other files - don't fail entire corpus load for one bad file

            self.files_loaded = loaded
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"Loaded corpus with {len(df)} total chunks (files_loaded={self.files_loaded})")
                self._log("info", f"Multi-file corpus loading completed", 
                         stage="corpus_loading", event="multi_file_completed", 
                         parquet_files_loaded=self.files_loaded, rows_loaded=len(df))
                return df
            print("Warning: no parquet files successfully loaded")
            return pd.DataFrame()
        except Exception as e:
            # For single parquet files, failures are critical - re-raise
            if self.corpus_path.endswith(".parquet"):
                raise
            # For multi-file scenarios, log and continue with empty corpus
            print(f"Error loading corpus: {e}")
            return pd.DataFrame()
    
    def _sample_parquet_files(self, files: List[str], limit: int = 50, mode: str = "strided") -> List[str]:
        """Sample parquet files using different strategies to avoid topic bias."""
        if len(files) <= limit:
            return files
        
        if mode == "strided":
            # Deterministic strided sampling for full coverage
            step = max(1, len(files) // limit)
            indices = list(range(0, len(files), step))[:limit]
            return [files[i] for i in indices]
        elif mode == "random":
            # Random sampling (for future use)
            import random
            return random.sample(files, limit)
        else:
            # Default: first N files (original behavior)
            return files[:limit]

    # ----------- public API used by pipeline -----------

    def bind_evidence(self, accepted_daydreams: List[Dict[str, Any]], paper_text: str) -> List[Dict[str, Any]]:
        """Bind (paper + corpus) evidence to accepted daydreams."""
        bound, total_quotes = [], 0
        print(f"Binding evidence for {len(accepted_daydreams)} daydreams...")
        self._log("info", f"Starting evidence binding", 
                 stage="evidence_binding", event="binding_start", 
                 daydreams_to_bind=len(accepted_daydreams))

        for i, d in enumerate(accepted_daydreams):
            self._update_heartbeat(i, "binding", len(bound))
            if total_quotes >= self.max_quotes_total:
                print(f"Reached quote budget ({self.max_quotes_total})")
                break

            ev = self._bind_single_daydream(d, paper_text)
            if self._has_required_evidence(ev):
                d2 = dict(d)
                d2["evidence"] = ev
                d2["theme_hint"] = self._generate_theme_hint(d)
                bound.append(d2)
                total_quotes += len(ev)
            else:
                print(f"Insufficient evidence for daydream {i+1}")

        print(f"Successfully bound evidence for {len(bound)} daydreams")
        print(f"Total quotes used: {total_quotes}/{self.max_quotes_total}")
        self._log("info", f"Evidence binding completed", 
                 stage="evidence_binding", event="binding_completed", 
                 bound_daydreams=len(bound), total_quotes=total_quotes, 
                 quote_budget=self.max_quotes_total)
        return bound

    def save_bound_daydreams(self, bound_daydreams: List[Dict[str, Any]], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for d in bound_daydreams:
                f.write(json.dumps(d) + "\n")

    def get_binding_statistics(self, bound_daydreams: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not bound_daydreams:
            return {}
        total_evidence = sum(len(d.get("evidence", [])) for d in bound_daydreams)
        paper_q = sum(sum(1 for e in d.get("evidence", []) if e.get("source") == "paper") for d in bound_daydreams)
        corpus_q = sum(sum(1 for e in d.get("evidence", []) if e.get("source") == "corpus") for d in bound_daydreams)
        supports = sum(sum(1 for e in d.get("evidence", []) if e.get("label") == "supports") for d in bound_daydreams)
        contradicts = sum(sum(1 for e in d.get("evidence", []) if e.get("label") == "contradicts") for d in bound_daydreams)

        return {
            "bound_daydreams": len(bound_daydreams),
            "total_evidence": total_evidence,
            "paper_quotes": paper_q,
            "corpus_quotes": corpus_q,
            "supports": supports,
            "contradicts": contradicts,
            "avg_evidence_per_daydream": total_evidence / max(1, len(bound_daydreams)),
        }

    # ----------- core binding -----------

    def _bind_single_daydream(self, daydream: Dict[str, Any], paper_text: str) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []

        # Paper quote (exact or fuzzy)
        pq = self._get_paper_anchor_quote(daydream, paper_text)
        if pq:
            evidence.append(pq)

        # Corpus quotes via 3-tier retrieval
        cq = self._get_corpus_quotes(daydream, max_quotes=2)
        evidence.extend(cq)

        # Track bound quotes metric
        self.metrics["bound_quotes"] += len(evidence)

        return evidence

    def write_metrics(self, output_dir: str = None):
        """Write binder metrics to JSON file in run directory."""
        try:
            import json
            import datetime
            
            # Add timestamp to metrics
            self.metrics["timestamp"] = datetime.datetime.now().isoformat()
            
            # Use run_dir by default, allow override for backwards compatibility
            target_dir = output_dir if output_dir is not None else self.run_dir
            metrics_path = os.path.join(target_dir, "binder_metrics.json")
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Binder metrics written to {metrics_path}")
            self._log("info", f"Binder metrics written successfully",
                     stage="metrics_writing", event="metrics_written", 
                     metrics_path=metrics_path)
            
        except Exception as e:
            logger.warning(f"Failed to write binder metrics: {e}")
            self._log("error", f"Failed to write binder metrics: {e}",
                     stage="metrics_writing", event="metrics_write_failed")
    
    def set_paper_id(self, paper_id: str):
        """Set paper_id for this binder instance and update logging context."""
        self.paper_id = paper_id
        
        # Update the logging formatter with the new paper_id
        if hasattr(self, '_log_formatter'):
            self._log_formatter.paper_id = paper_id
            
        self._log("info", f"Paper ID set for binder context", 
                 stage="initialization", event="paper_id_set", 
                 paper_id=paper_id)

    # ----------- paper anchor extraction -----------

    def _get_paper_anchor_quote(self, daydream: Dict[str, Any], paper_text: str) -> Optional[Dict[str, Any]]:
        anchor = (daydream.get("paper_anchor") or "").strip()
        concept = (daydream.get("paper_concept") or "").strip()

        if not anchor and not concept:
            return None

        low = paper_text.lower()
        if anchor and anchor.lower() in low:
            start = low.index(anchor.lower())
            q = self._extract_quote_window(paper_text, start, target_words=80)
            if q:
                return {
                    "source": "paper",
                    "label": "supports",
                    "citation": f"(paper@{q['char_start']}-{q['char_end']})",
                    "quote": q["quote"],
                }

        # Fuzzy fallback: best matching sentence to (anchor or concept)
        sentences = self._split_into_sentences(paper_text)
        if not sentences:
            return None

        key = (anchor or concept).lower()
        from difflib import SequenceMatcher
        best_i, best_score = -1, 0.0
        for i, s in enumerate(sentences):
            sc = SequenceMatcher(None, key, s.lower()).ratio()
            if sc > best_score:
                best_score, best_i = sc, i

        if best_score >= 0.55:
            # compute char start
            char_start = 0
            for j, s in enumerate(sentences):
                if j == best_i:
                    break
                # +1 rough separator
                char_start += len(s) + 1
            q = self._extract_quote_window(paper_text, char_start, target_words=80)
            if q:
                return {
                    "source": "paper",
                    "label": "supports",
                    "citation": f"(paper@{q['char_start']}-{q['char_end']})",
                    "quote": q["quote"],
                }
        return None

    def _extract_quote_window(self, text: str, char_start: int, target_words: int = 80) -> Optional[Dict[str, str]]:
        words = re.findall(r"\b\w+\b", text)
        # Build indices of word starts
        word_starts, pos = [], 0
        for w in words:
            wp = text.find(w, pos)
            word_starts.append(wp)
            pos = wp + len(w)

        # Find which word char_start lands in
        start_word_idx = 0
        for i, wp in enumerate(word_starts):
            if wp <= char_start:
                start_word_idx = i
            else:
                break

        half = target_words // 2
        si = max(0, start_word_idx - half)
        ei = min(len(words), start_word_idx + half)

        if si >= ei:
            return None

        start_char = word_starts[si] if si < len(word_starts) else 0
        end_char = (word_starts[ei] if ei < len(word_starts) else word_starts[-1] + len(words[-1]))
        quote = text[start_char:end_char].strip()

        if self.min_quote_length <= len(quote) <= self.max_quote_length * 2:
            return {"quote": quote, "char_start": start_char, "char_end": end_char}
        return None

    # ----------- semantic indexing -----------
    
    def _compute_corpus_fingerprint(self) -> str:
        """Compute a fingerprint of the loaded corpus for cache validation."""
        try:
            # Create hash from corpus content and metadata
            hash_input = []
            
            # Add corpus path and modification time if single file
            if self.corpus_path.endswith(".parquet"):
                if os.path.exists(self.corpus_path):
                    stat = os.stat(self.corpus_path)
                    hash_input.append(f"file:{self.corpus_path}:mtime:{stat.st_mtime}:size:{stat.st_size}")
                else:
                    hash_input.append(f"file:{self.corpus_path}:missing")
            else:
                # For directory, include all loaded parquet files
                files = self._discover_parquets(self.corpus_path)
                default_max_files = self.parq_cfg.get("max_files", 200)
                max_files = min(default_max_files, len(files))
                selected_files = self._sample_parquet_files(files, max_files, "strided")
                
                for fp in selected_files:
                    if os.path.exists(fp):
                        stat = os.stat(fp)
                        hash_input.append(f"file:{fp}:mtime:{stat.st_mtime}:size:{stat.st_size}")
                    else:
                        hash_input.append(f"file:{fp}:missing")
            
            # Add corpus data properties  
            hash_input.append(f"rows:{len(self.corpus_df)}")
            hash_input.append(f"cols:{list(self.corpus_df.columns)}")
            
            # Add semantic indexing config that affects index structure
            semantic_cfg = self.config.get('binder', {}).get('semantic', {})
            relevant_config = {
                'model_name': semantic_cfg.get('model_name'),
                'normalize_embeddings': semantic_cfg.get('normalize_embeddings', True),
                'max_docs': semantic_cfg.get('max_docs', 20000)  # Will be changed in P0.4
            }
            hash_input.append(f"config:{relevant_config}")
            
            # Create SHA256 hash
            fingerprint_str = "|".join(sorted(hash_input))
            return hashlib.sha256(fingerprint_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to compute corpus fingerprint: {e}")
            return "error_" + str(int(time.time()))
    
    def _get_index_cache_path(self, fingerprint: str) -> tuple[Path, Path, Path]:
        """Get paths for cached index files."""
        cache_dir = Path(self.semantic_cfg.get("index_cache_dir", "./.binder_index"))
        cache_dir.mkdir(exist_ok=True)
        
        index_path = cache_dir / f"faiss_index_{fingerprint[:16]}.index"
        doc_ids_path = cache_dir / f"doc_ids_{fingerprint[:16]}.npy"
        metadata_path = cache_dir / f"metadata_{fingerprint[:16]}.json"
        
        return index_path, doc_ids_path, metadata_path
    
    def _load_cached_index(self, fingerprint: str) -> bool:
        """Load cached FAISS index if available and valid."""
        try:
            index_path, doc_ids_path, metadata_path = self._get_index_cache_path(fingerprint)
            
            # Check if all required files exist
            if not (index_path.exists() and doc_ids_path.exists() and metadata_path.exists()):
                return False
                
            # Load metadata and validate
            with open(metadata_path) as f:
                metadata = json.load(f)
                
            if metadata.get('fingerprint') != fingerprint:
                logger.warning("Index fingerprint mismatch, rebuilding")
                return False
                
            # Load FAISS index
            import faiss
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load doc_ids  
            self.faiss_ids = np.load(doc_ids_path, allow_pickle=True)
            
            logger.info(f"Loaded cached semantic index: {len(self.faiss_ids)} docs, fingerprint: {fingerprint[:16]}")
            self._log("info", "Cached semantic index loaded successfully",
                     stage="semantic_indexing", event="cache_loaded",
                     indexed_docs=len(self.faiss_ids),
                     fingerprint=fingerprint[:16])
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            return False
    
    def _save_index_to_cache(self, fingerprint: str):
        """Save FAISS index to cache for reuse."""
        if not self.faiss_index or self.faiss_ids is None:
            return
            
        try:
            index_path, doc_ids_path, metadata_path = self._get_index_cache_path(fingerprint)
            
            # Save FAISS index
            import faiss
            faiss.write_index(self.faiss_index, str(index_path))
            
            # Save doc_ids
            np.save(doc_ids_path, self.faiss_ids)
            
            # Save metadata
            metadata = {
                'fingerprint': fingerprint,
                'timestamp': time.time(),
                'indexed_docs': len(self.faiss_ids),
                'vector_dim': self.faiss_index.d,
                'model_name': self.semantic_cfg.get('model_name'),
                'created_by': 'DDLEvidenceBinder'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved semantic index to cache: fingerprint {fingerprint[:16]}")
            self._log("info", "Semantic index saved to cache", 
                     stage="semantic_indexing", event="cache_saved",
                     fingerprint=fingerprint[:16],
                     cache_path=str(index_path))
                     
        except Exception as e:
            logger.warning(f"Failed to save index to cache: {e}")
    
    def _build_semantic_index(self):
        """Build FAISS index for semantic similarity search with persistence and full corpus support."""
        if not self.embedding_model or self.corpus_df.empty:
            return
            
        try:
            # Compute corpus fingerprint for cache validation
            fingerprint = self._compute_corpus_fingerprint()
            
            # Try to load from cache first
            if self._load_cached_index(fingerprint):
                return  # Successfully loaded from cache
            
            # Cache miss or invalid - build new index
            logger.info("Building new semantic index (cache miss or invalid)...")
            
            # P0.4: Use full loaded corpus instead of 20k sample
            # With 512GB RAM, we can handle 5-10M docs (20-40GB for embeddings)
            max_docs = self.semantic_cfg.get("max_docs", len(self.corpus_df))  # Use full corpus by default
            
            if max_docs >= len(self.corpus_df):
                # Use full corpus
                df_to_index = self.corpus_df
                logger.info(f"Indexing full corpus: {len(df_to_index)} documents")
            else:
                # Use sampling only if explicitly limited
                df_to_index = self.corpus_df.sample(n=min(max_docs, len(self.corpus_df)), random_state=42)
                logger.warning(f"Indexing sampled corpus: {len(df_to_index)}/{len(self.corpus_df)} documents (consider increasing max_docs)")
            
            # Extract texts and normalize
            texts = df_to_index["text"].astype(str).tolist()
            doc_ids = df_to_index["doc_id"].astype(str).tolist()
            
            self._log("info", f"Starting semantic index build", 
                     stage="semantic_indexing", event="index_build_start", 
                     documents_to_index=len(texts), 
                     total_corpus_docs=len(self.corpus_df),
                     full_corpus_indexing=(max_docs >= len(self.corpus_df)),
                     fingerprint=fingerprint[:16])
            
            # Encode texts in batches with progress tracking
            normalize_embeddings = self.semantic_cfg.get("normalize_embeddings", True)
            batch_size = self.semantic_cfg.get("batch_size", 128)
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            embeddings_list = []
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Progress heartbeat during embedding
                self._progress_heartbeat("semantic_encoding", batch_idx + 1, total_batches, 
                                       {"docs_encoded": end_idx, "batch_size": len(batch_texts)})
                
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings_list.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(embeddings_list)
            
            # Build FAISS index
            dim = embeddings.shape[1]
            import faiss
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
            self.faiss_index.add(embeddings.astype("float32"))
            self.faiss_ids = np.array(doc_ids, dtype=object)
            
            logger.info(f"Semantic index built: {dim}D vectors, {len(doc_ids)} docs")
            self._log("info", f"Semantic index build completed", 
                     stage="semantic_indexing", event="index_build_completed", 
                     vector_dim=dim, indexed_docs=len(doc_ids))
            
            # Save to cache for future runs
            self._save_index_to_cache(fingerprint)
            
        except Exception as e:
            logger.warning(f"Failed to build semantic index: {e}")
            self.faiss_index = None
            self.faiss_ids = None
    
    def _semantic_search(self, query_text: str, topk: int = 200) -> List[Tuple[str, float]]:
        """Search semantic index for similar documents."""
        if not self.faiss_index or not self.embedding_model:
            return []
            
        try:
            # Encode query
            normalize_embeddings = self.semantic_cfg.get("normalize_embeddings", True)
            query_emb = self.embedding_model.encode(
                [query_text], 
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )[0].astype("float32")
            
            # Search index
            scores, indices = self.faiss_index.search(
                np.expand_dims(query_emb, 0), 
                min(topk, len(self.faiss_ids))
            )
            
            # Return doc_id, score pairs
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i >= 0 and i < len(self.faiss_ids):  # Valid index
                    results.append((self.faiss_ids[i], float(score)))
                    
            return results
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    def _hybrid_rank(self, query: str, bm25_candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine BM25 and semantic scores for hybrid ranking."""
        if not self.embedding_model:
            return bm25_candidates
            
        # Get semantic candidates
        sem_topk = self.semantic_cfg.get("topk", 200)
        sem_hits = self._semantic_search(query, sem_topk)
        
        # Build score maps
        bm25_map = {doc_id: score for doc_id, score in bm25_candidates}
        sem_map = {doc_id: score for doc_id, score in sem_hits}
        
        # Combine scores (union of candidates)
        combined = {}
        for doc_id, score in bm25_candidates:
            combined[doc_id] = {"bm25": score, "sem": 0.0}
        for doc_id, score in sem_hits:
            if doc_id not in combined:
                combined[doc_id] = {"bm25": 0.0, "sem": score}
            else:
                combined[doc_id]["sem"] = max(combined[doc_id]["sem"], score)
        
        # Weight and score
        w_sem = self.semantic_cfg.get("weight_semantic", 0.65)
        w_bm25 = self.semantic_cfg.get("weight_bm25", 0.35)
        
        scored = []
        for doc_id, scores in combined.items():
            hybrid_score = w_sem * scores["sem"] + w_bm25 * scores["bm25"]
            scored.append((doc_id, hybrid_score))
            
        # Sort by hybrid score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ----------- retrieval tiers -----------

    def _get_corpus_quotes(self, daydream: Dict[str, Any], max_quotes: int = 2) -> List[Dict[str, Any]]:
        if self.corpus_df.empty:
            return []

        paper_concept = (daydream.get("paper_concept") or "").strip()
        corpus_concept = (daydream.get("corpus_concept") or "").strip()
        hypothesis = (daydream.get("hypothesis") or "").strip()

        quotes: List[Dict[str, Any]] = []
        stats = {"tierA": 0, "tierB": 0, "tierC": 0, "supports": 0, "contradicts": 0, "neutrals": 0, "nli_fails": 0}

        # Tier A: strict (both concepts present)
        tier_a = self._retrieve_tier_a(paper_concept, corpus_concept, hypothesis, k=120)
        for c in tier_a:
            c["retrieval_tier"] = "A"
        stats["tierA"] = len(tier_a)
        if tier_a:
            labeled_a, _ = self._nli_batch_label_with_budget(tier_a, hypothesis, timeout_s=self.nli_timeout_s)
            quotes.extend([q for q in labeled_a if q["label"] in ("supports", "contradicts")])
            self._update_nli_stats(labeled_a, stats)
            if len(quotes) >= max_quotes:
                self._log_binding_stats(**stats)
                return quotes[:max_quotes]

        # Tier B: relaxed (one concept + hint from hypothesis head)
        if len(quotes) < max_quotes:
            tier_b = self._retrieve_tier_b(paper_concept, corpus_concept, hypothesis, k=200)
            for c in tier_b:
                c["retrieval_tier"] = "B"
            stats["tierB"] = len(tier_b)
            if tier_b:
                labeled_b, _ = self._nli_batch_label_with_budget(tier_b, hypothesis, timeout_s=self.nli_timeout_s)
                quotes.extend([q for q in labeled_b if q["label"] in ("supports", "contradicts")])
                self._update_nli_stats(labeled_b, stats)
                if len(quotes) >= max_quotes:
                    self._log_binding_stats(**stats)
                    return quotes[:max_quotes]

        # Tier C: semantic-lite (lexical overlap only; mark as weak)
        if len(quotes) < max_quotes:
            tier_c = self._retrieve_tier_c(paper_concept, corpus_concept, hypothesis, k=300)
            for c in tier_c:
                c["retrieval_tier"] = "C"
            stats["tierC"] = len(tier_c)
            if tier_c:
                labeled_c, _ = self._nli_batch_label_with_budget(tier_c, hypothesis, timeout_s=self.nli_timeout_s)
                for q in labeled_c:
                    if q["label"] in ("supports", "contradicts"):
                        q["evidence_strength"] = "weak"
                        quotes.append(q)
                self._update_nli_stats(labeled_c, stats)

        # Sort strong first, then supports, then score
        quotes.sort(
            key=lambda x: (
                x.get("evidence_strength", "strong") == "strong",
                x["label"] == "supports",
                float(x.get("score", 0.0)),
            ),
            reverse=True,
        )
        self._log_binding_stats(**stats)
        return quotes[:max_quotes]

    def _expand_concept_aliases(self, concept: str) -> List[str]:
        """Expand concept with 2-3 aliases for better Tier-A matching."""
        concept_lower = concept.lower()
        aliases = [concept]
        
        # Simple alias expansion based on common patterns
        alias_map = {
            'honest': ['honesty', 'truthful', 'truth'],
            'ai': ['artificial intelligence', 'machine learning', 'ml'],
            'alignment': ['aligned', 'safe ai', 'ai safety'],
            'reasoning': ['logic', 'inference', 'rational'],
            'language model': ['llm', 'large language model', 'gpt'],
            'transformer': ['attention', 'neural network', 'deep learning'],
            'sparse autoencoder': ['sae', 'autoencoder', 'sparse coding'],
            'mechanistic interpretability': ['mech interp', 'interpretability', 'explainable ai'],
            'deception': ['lying', 'dishonest', 'misleading'],
            'capability': ['skill', 'ability', 'competence'],
            'oversight': ['supervision', 'monitoring', 'control']
        }
        
        for key, vals in alias_map.items():
            if key in concept_lower:
                aliases.extend([v for v in vals if v not in aliases])
                break
        
        # Add partial matches and plurals
        if not concept.endswith('s') and len(concept) > 4:
            aliases.append(concept + 's')
        if concept.endswith('s') and len(concept) > 4:
            aliases.append(concept[:-1])
            
        return aliases[:3]  # Limit to 3 total

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _retrieve_tier_a_semantic(self, paper_concept: str, corpus_concept: str, hypothesis: str, k: int) -> List[Dict[str, Any]]:
        """Semantic similarity-based Tier-A retrieval as fallback to lexical."""
        if not self.embedding_model:
            return []
        
        try:
            # Compute embeddings for concepts
            paper_embedding = self.embedding_model.encode(paper_concept)
            corpus_embedding = self.embedding_model.encode(corpus_concept)
            
            candidates = []
            
            # Sample chunks for semantic similarity (limit to prevent memory issues)
            chunk_sample = self.corpus_df.sample(n=min(1000, len(self.corpus_df)), random_state=42)
            
            for idx, chunk in chunk_sample.iterrows():
                chunk_text = str(chunk.get('text', ''))[:500]  # Limit text length
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                    
                # Compute chunk embedding
                chunk_embedding = self.embedding_model.encode(chunk_text)
                
                # Calculate similarity to both concepts
                paper_sim = self._cosine_similarity(paper_embedding, chunk_embedding)
                corpus_sim = self._cosine_similarity(corpus_embedding, chunk_embedding)
                
                # Require moderate similarity to both concepts
                if paper_sim > 0.5 and corpus_sim > 0.5:
                    combined_score = paper_sim * corpus_sim
                    candidates.append({
                        'text': chunk_text,
                        'score': combined_score,
                        'paper_similarity': paper_sim,
                        'corpus_similarity': corpus_sim,
                        'author': chunk.get('author', 'Unknown'),
                        'year': chunk.get('year', 'Unknown'),
                        'doc_id': chunk.get('doc_id', 'Unknown'),
                        'retrieval_tier': 'A-semantic'
                    })
            
            # Sort by combined score and return top-k
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:k]
            
        except Exception as e:
            logger.warning(f"Tier-A-Semantic failed: {e}")
            return []

    def _retrieve_tier_a(self, paper_concept: str, corpus_concept: str, hypothesis: str, k: int) -> List[Dict[str, Any]]:
        # Require both concepts present (AND) - use aliases for better matching
        candidates: List[Dict[str, Any]] = []
        
        # Expand both concepts with aliases
        paper_aliases = self._expand_concept_aliases(paper_concept)
        corpus_aliases = self._expand_concept_aliases(corpus_concept)
        
        # Build queries with concept combinations
        queries = []
        for pa in paper_aliases[:2]:  # Top 2 paper aliases
            for ca in corpus_aliases[:2]:  # Top 2 corpus aliases
                queries.append(f"{pa} {ca}")
        queries = list(dict.fromkeys(queries))  # Deduplicate
        
        # Must contain at least one alias from each concept
        must_paper = [t.lower() for t in paper_aliases]
        must_corpus = [t.lower() for t in corpus_aliases]

        for q in queries:
            if not q.strip():
                continue
            part = self._retrieve_chunks_with_filters(
                q, 
                top_k=max(1, k // len(queries)),
                must_contain_paper=must_paper,
                must_contain_corpus=must_corpus,
                phrase_boost=True
            )
            candidates.extend(part)
        result = self._deduplicate_candidates(candidates)[:k]
        
        # If lexical matching fails, try semantic similarity as fallback
        if len(result) == 0 and self.embedding_model:
            semantic_result = self._retrieve_tier_a_semantic(paper_concept, corpus_concept, hypothesis, k)
            if semantic_result:
                return semantic_result
        
        return result

    def _retrieve_tier_b(self, paper_concept: str, corpus_concept: str, hypothesis: str, k: int) -> List[Dict[str, Any]]:
        # Require at least one concept present (OR), add hypothesis head
        candidates: List[Dict[str, Any]] = []
        hyp_head = hypothesis[:100]
        queries = list(dict.fromkeys([f"{paper_concept} {corpus_concept}", paper_concept, corpus_concept, hyp_head]))
        any_terms = [t.lower() for t in [paper_concept, corpus_concept] if t]

        for q in queries:
            if not q.strip():
                continue
            part = self._retrieve_chunks_with_filters(q, top_k=max(1, k // len(queries)),
                                                      must_contain_any=any_terms,
                                                      phrase_boost=False)
            candidates.extend(part)
        return self._deduplicate_candidates(candidates)[:k]

    def _retrieve_tier_c(self, paper_concept: str, corpus_concept: str, hypothesis: str, k: int) -> List[Dict[str, Any]]:
        # No lexical filters; lexical overlap scoring only
        candidates: List[Dict[str, Any]] = []
        queries = list(dict.fromkeys([hypothesis, f"{paper_concept} {corpus_concept}", paper_concept, corpus_concept]))
        for q in queries:
            if not q.strip():
                continue
            part = self._retrieve_chunks(q, top_k=max(1, k // len(queries)))
            candidates.extend(part)
        return self._deduplicate_candidates(candidates)[:k]

    # ----------- retrieval helpers -----------

    def _retrieve_chunks_with_filters(self,
                                      query: str,
                                      top_k: int,
                                      must_contain: Optional[List[str]] = None,
                                      must_contain_any: Optional[List[str]] = None,
                                      must_contain_paper: Optional[List[str]] = None,
                                      must_contain_corpus: Optional[List[str]] = None,
                                      phrase_boost: bool = False) -> List[Dict[str, Any]]:
        if self.corpus_df.empty:
            return []

        base = self._retrieve_chunks(query, top_k=top_k * 3)
        filtered: List[Dict[str, Any]] = []
        must = [t for t in (must_contain or []) if t]
        any_terms = [t for t in (must_contain_any or []) if t]
        paper_aliases = [t.lower() for t in (must_contain_paper or []) if t]
        corpus_aliases = [t.lower() for t in (must_contain_corpus or []) if t]

        for c in base:
            t = c["text"].lower()
            
            # Drop overly long candidates (>3,000 chars) - too long for good NLI
            if len(t) > 3000:
                continue

            # AND filter (original)
            if must and (not all(term in t for term in must)):
                continue

            # OR filter (original)
            if any_terms and (not any(term in t for term in any_terms)):
                continue
            
            # Paper AND Corpus alias filter (for Tier-A)
            if paper_aliases and corpus_aliases:
                has_paper = any(alias in t for alias in paper_aliases)
                has_corpus = any(alias in t for alias in corpus_aliases)
                if not (has_paper and has_corpus):
                    continue

            # Phrase boost (enhanced for alias matching)
            boost_applied = False
            if phrase_boost:
                if must and all(term in t for term in must):
                    c["score"] = float(c.get("score", 0.5)) * 1.5
                    boost_applied = True
                elif paper_aliases and corpus_aliases:
                    has_paper = any(alias in t for alias in paper_aliases)
                    has_corpus = any(alias in t for alias in corpus_aliases)
                    if has_paper and has_corpus:
                        c["score"] = float(c.get("score", 0.5)) * 1.8  # Higher boost for alias matches
                        boost_applied = True

            filtered.append(c)

        filtered.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return filtered[:top_k]

    def _retrieve_chunks(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """BM25-lite retrieval with TF-IDF scoring for better relevance."""
        if self.corpus_df.empty:
            return []
        
        import math
        qtokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3]
        if not qtokens:
            return []

        matches: List[Dict[str, Any]] = []
        scan_cap = min(len(self.corpus_df), self.scan_cap)
        
        # Pre-compute document frequency for IDF calculation
        doc_freq = {}
        N = scan_cap  # Total number of documents
        
        # First pass: calculate document frequencies
        for i, (idx, row) in enumerate(self.corpus_df.iloc[:scan_cap].iterrows()):
            # Progress heartbeat during document frequency calculation
            self._progress_heartbeat("bm25_docfreq", i + 1, scan_cap, 
                                   {"query": query[:50], "scan_limit": scan_cap})
            
            text = str(row["text"]).lower()
            tokens = [t for t in re.findall(r"\w+", text) if len(t) >= 3]
            seen_terms = set()
            for term in qtokens:
                if term in tokens and term not in seen_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
                    seen_terms.add(term)
        
        # Second pass: calculate BM25-lite scores
        for i, (idx, row) in enumerate(self.corpus_df.iloc[:scan_cap].iterrows()):
            # Progress heartbeat during BM25 scoring
            self._progress_heartbeat("bm25_scoring", i + 1, scan_cap, 
                                   {"candidates_found": len(matches)})
            
            text = str(row["text"]).lower()
            tokens = [t for t in re.findall(r"\w+", text) if len(t) >= 3]
            
            score = 0.0
            for term in qtokens:
                if doc_freq.get(term, 0) == 0:
                    continue
                    
                # Term frequency in document
                tf = sum(1 for t in tokens if t == term)
                if tf == 0:
                    continue
                
                # Inverse document frequency
                idf = math.log((N - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
                
                # BM25-lite score (simplified, no document length normalization)
                score += tf * idf
            
            if score > 0:
                matches.append({
                    "text": row["text"],
                    "author": row.get("author", "Unknown"),
                    "year": row.get("year", 2020),
                    "title": row.get("title", "Untitled"),
                    "doc_id": row.get("doc_id", f"row_{idx}"),
                    "score": float(score),
                })

        # Update BM25 candidates metric
        self.metrics["bm25_candidates_considered"] += len(matches)
        
        # Apply hybrid ranking if semantic indexing is enabled
        if self.faiss_index is not None and matches:
            # Convert to doc_id, score tuples for hybrid ranking
            bm25_candidates = [(m["doc_id"], m["score"]) for m in matches]
            
            # Get hybrid rankings (includes both BM25 + semantic-only candidates)
            hybrid_candidates = self._hybrid_rank(query, bm25_candidates)
            
            # Create doc_id to match mapping for reordering  
            doc_id_to_match = {m["doc_id"]: m for m in matches}
            
            # Preserve original BM25 scores before overwriting
            for m in matches:
                m["original_bm25_score"] = m["score"]
            
            # Process hybrid candidates (includes BM25 + semantic-only)
            reordered_matches = []
            semantic_only_cap = 50  # Limit semantic-only additions to control NLI cost
            semantic_only_added = 0
            
            for doc_id, hybrid_score in hybrid_candidates:
                if doc_id in doc_id_to_match:
                    # Existing BM25 candidate: reorder with hybrid score
                    match = doc_id_to_match[doc_id]
                    match["score"] = hybrid_score  # Hybrid score for ranking
                    match["hybrid_score"] = hybrid_score  # Also store as explicit field
                    reordered_matches.append(match)
                else:
                    # Semantic-only candidate: create new match from corpus_df
                    if semantic_only_added >= semantic_only_cap:
                        continue
                    
                    # Find the row in corpus_df by doc_id
                    matching_rows = self.corpus_df[self.corpus_df.get('doc_id', pd.Series(dtype=object)) == doc_id]
                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        semantic_match = {
                            "text": row["text"],
                            "author": row.get("author", "Unknown"),
                            "year": row.get("year", 2020),
                            "title": row.get("title", "Untitled"),
                            "doc_id": doc_id,
                            "score": hybrid_score,
                            "hybrid_score": hybrid_score,
                            "original_bm25_score": 0.0,  # Not found in BM25
                            "semantic_only": True  # Mark as semantic-only discovery
                        }
                        reordered_matches.append(semantic_match)
                        semantic_only_added += 1
            
            matches = reordered_matches
            
            # Log semantic enhancement stats
            self._log("info", f"Hybrid ranking applied", 
                     stage="hybrid_ranking", event="reranking_completed",
                     original_bm25_candidates=len(bm25_candidates),
                     total_hybrid_candidates=len(hybrid_candidates), 
                     semantic_only_added=semantic_only_added,
                     final_matches=len(matches))
        else:
            matches.sort(key=lambda x: x["score"], reverse=True)
            
        return matches[:top_k]

    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, uniq = set(), []
        for c in candidates:
            key = (c.get("doc_id", ""), c.get("text", "")[:120])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    # ----------- NLI labeling -----------

    def _nli_batch_label_with_budget(self,
                                     candidates: List[Dict[str, Any]],
                                     hypothesis: str,
                                     timeout_s: int = 4) -> Tuple[List[Dict[str, Any]], int]:
        """Label candidates with NLI; include fallback + tier stamp through to quotes."""
        labeled: List[Dict[str, Any]] = []
        used_calls = 0

        for c in candidates:
            label = self._nli_label_with_fallback(hypothesis, c["text"], timeout_s)
            nli_method = "nli"  # we don't expose whether fallback used inside; see _lexical_fallback_label

            # Update NLI metrics
            self.metrics["nli_pairs_evaluated"] += 1

            # minimal quote extraction
            q = self._extract_quote(c["text"])
            labeled.append({
                "source": "corpus",
                "label": label,
                "citation": f"({c.get('author', 'Unknown')}, {c.get('year', 'Unknown')}; {c.get('doc_id', 'Unknown')})",
                "quote": q,
                "score": float(c.get("score", 0.5)),
                "retrieval_tier": c.get("retrieval_tier", "?")
            })
            used_calls += 1
        return labeled, used_calls

    def _truncate_words(self, s: str, n: int) -> str:
        toks = re.findall(r"\w+|\S", s)
        if len(toks) <= n: 
            return s
        out = " ".join(toks[:n])
        return out

    def _clean_quote(self, s: str) -> str:
        # strip urls/page headers/boilerplate that explode tokens
        s = re.sub(r'https?://\S+', '', s)
        s = re.sub(r'Page \d+[: ]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _nli_label_with_fallback(self, claim_text: str, quote: str, timeout_s: int = 4) -> str:
        """Ask local LLM for NLI label; on error or junk, use lexical fallback."""
        # Critical: truncate inputs to prevent context overflow
        claim = self._truncate_words(claim_text, 80)   # keep claim short & focused
        quote = self._clean_quote(self._truncate_words(quote, 120))  # 100–120 tokens max
        
        system_prompt = (
            "Label the relationship between the QUOTE and the CLAIM.\n\n"
            "Return exactly one word:\n"
            '- "supports" if the quote provides evidence supporting the claim\n'
            '- "contradicts" if the quote provides evidence against the claim\n'
            '- "neutral" if neither\n'
        )
        user_prompt = f"CLAIM: {claim}\n\nQUOTE: {quote}\n\nRelationship:"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=16,
                timeout=timeout_s
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if content in ("supports", "contradicts", "neutral"):
                return content
            if "support" in content:
                return "supports"
            if "contradict" in content:
                return "contradicts"
            return "neutral"
        except Exception as e:
            print(f"NLI timeout/error ({timeout_s}s): {e}")
            # Track timeout/error in metrics
            self.metrics["nli_timeouts"] += 1
            self._log("warning", f"NLI timeout/error during labeling", 
                     stage="nli_labeling", event="nli_timeout", 
                     timeout_s=timeout_s, nli_timeouts=self.metrics["nli_timeouts"])
            return self._lexical_fallback_label(claim_text, quote)

    def _lexical_fallback_label(self, claim_text: str, quote: str) -> str:
        """Heuristic label if LLM/NLI is unavailable/unreliable."""
        ql = quote.lower()
        contra = [" not ", "fails", "unable", "contradict", "however", " but ", " although ", " nevertheless ",
                  " despite ", " except ", " unless ", "wrong", "incorrect", "false"]
        if any(w in ql for w in contra):
            return "contradicts"
        supp = [" show", " evidence", " support", "consistent", " replicate", " confirm", " validate",
                " demonstrate", " prove", " establish", " indicate", " suggest"]
        if any(w in ql for w in supp):
            return "supports"
        return "neutral"

    def _update_nli_stats(self, labeled_quotes: List[Dict[str, Any]], stats: Dict[str, int]):
        for q in labeled_quotes:
            lab = q.get("label", "neutral")
            if lab == "supports":
                stats["supports"] += 1
            elif lab == "contradicts":
                stats["contradicts"] += 1
            elif lab == "neutral":
                stats["neutrals"] += 1
            else:
                stats["nli_fails"] += 1

    def _log_binding_stats(self, tierA: int, tierB: int, tierC: int,
                           supports: int, contradicts: int, neutrals: int, nli_fails: int):
        print(f"[Binder] tierA:{tierA} | tierB:{tierB} | tierC:{tierC} | "
              f"supports:{supports} opposes:{contradicts} neutrals:{neutrals} | nli_fail:{nli_fails}")

    # ----------- utilities -----------

    def _extract_quote(self, text: str) -> str:
        """Take a well‑formed sentence or merge short ones to ~max_quote_length."""
        sents = self._split_into_sentences(text)
        if not sents:
            return text[: self.max_quote_length].strip()
        for s in sents:
            if self.min_quote_length <= len(s) <= self.max_quote_length:
                return s.strip()
        combined = ""
        for s in sents:
            if len((combined + " " + s).strip()) <= self.max_quote_length:
                combined = (combined + " " + s).strip()
            else:
                break
        if len(combined) >= self.min_quote_length:
            return combined
        return text[: self.max_quote_length].strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter
        parts = re.split(r"[.!?]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _has_required_evidence(self, evidence: List[Dict[str, Any]]) -> bool:
        paper_q = [e for e in evidence if e.get("source") == "paper"]
        corpus_q = [e for e in evidence if e.get("source") == "corpus"]
        return len(paper_q) >= 1 and len(corpus_q) >= 1

    def _generate_theme_hint(self, daydream: Dict[str, Any]) -> str:
        pc = (daydream.get("paper_concept") or "").strip()
        cc = (daydream.get("corpus_concept") or "").strip()
        if pc and cc:
            return f"{pc} & {cc}"
        return pc or cc or "General"

    def _update_heartbeat(self, pair_idx: int, tier: str, kept: int):
        """Write heartbeat file and log heartbeat events every 30 seconds."""
        now = time.time()
        last_file = getattr(self, "last_heartbeat", 0.0) or 0.0
        last_log = getattr(self, "last_heartbeat_log", 0.0) or 0.0
        
        # Update heartbeat file every 5 seconds (existing behavior)
        if now - last_file >= 5:
            payload = {"ts": now, "pair_idx": pair_idx, "stage": tier, "kept": kept, "files_loaded": self.files_loaded}
            try:
                with open(self.heartbeat_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                self.last_heartbeat = now
            except Exception:
                # don't crash on heartbeat
                pass
        
        # Log heartbeat event every 30 seconds for monitoring
        if now - last_log >= 30:
            self._log("info", f"Processing heartbeat", 
                     stage=tier, event="heartbeat", 
                     pairs_processed=pair_idx, 
                     parquet_files_loaded=self.files_loaded,
                     nli_timeouts=self.metrics.get("nli_timeouts", 0),
                     bound_items=kept)
            self.last_heartbeat_log = now

    def _progress_heartbeat(self, stage: str, current: int, total: int, extra_info: dict = None):
        """Log progress heartbeat for long-running operations."""
        now = time.time()
        last_progress = getattr(self, f"_last_progress_{stage}", 0.0)
        
        # Log progress every 30 seconds
        if now - last_progress >= 30:
            progress_pct = round((current / total * 100), 1) if total > 0 else 0
            
            log_data = {
                "stage": stage,
                "event": "heartbeat",
                "progress_pct": progress_pct,
                "items_processed": current,
                "total_items": total,
                "parquet_files_loaded": self.files_loaded
            }
            
            if extra_info:
                log_data.update(extra_info)
                
            self._log("info", f"Progress update for {stage}", **log_data)
            setattr(self, f"_last_progress_{stage}", now)


# ----------- local smoke test -----------

if __name__ == "__main__":
    binder = DDLEvidenceBinder("chunked_corpus/")
    # Mock daydream
    d = {
        "paper_concept": "sparse autoencoder",
        "corpus_concept": "attention mechanism",
        "hypothesis": "Sparse autoencoders could be combined with attention mechanisms to expose and manipulate structured feature directions that mediate control tokens.",
        "paper_anchor": "sparse autoencoder",
    }
    paper = "Sparse autoencoders learn compressed representations by forcing most neurons to be inactive, creating interpretable features that can steer model behavior in controlled ways."

    ev = binder._bind_single_daydream(d, paper)
    print(f"Found {len(ev)} evidence items")
    for e in ev:
        print(f"- {e['source']} ({e.get('label','?')} | {e.get('retrieval_tier','paper')}): {e['quote'][:80]}...")
