# DDL V8: Hybrid Model System

## Overview
Version 8 implements a hybrid approach combining two specialized models for optimal performance and speed:

- **🧠 Qwen 235B-A22B-Thinking (LM Studio)**: Creative tasks requiring deep reasoning
- **⚡ Qwen 30B (Ollama)**: Evidence binding for 3-5x speed improvement

## Quick Start

```bash
cd /Users/willkirby/scrape\ 2/LW_scrape/v8

# Preflight check (recommended first step)
python3 scripts/preflight.py

# Test run (3 pairs, default rows acceptance)
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 3

# Production run (200 pairs) 
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 200

# Custom acceptance criteria
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 200 \
  --accept-by docs --docs-min 25000

# Legacy file-based acceptance  
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 200 \
  --accept-by files --files-min 600
```

## Architecture

### Hybrid Model Distribution
| Stage | Model | Port | Purpose |
|-------|-------|------|---------|
| Paper Concept Extraction | Qwen 235B | 1234 | Deep semantic understanding |
| Daydream Generation | Qwen 235B | 1234 | Creative hypothesis generation |
| Critic Filtering | Qwen 235B | 1234 | Quality evaluation |
| **Evidence Binding** | **Qwen 30B** | **11434** | **Fast evidence retrieval** |
| Final Composition | Qwen 235B | 1234 | Sophisticated writing |

### Performance Benefits
- **Speed**: Evidence binding ~3-5x faster with 30B model
- **Quality**: Maintains 235B thinking power for creative analysis
- **Parallel**: LM Studio + Ollama run simultaneously
- **Memory**: Efficient resource utilization

## Prerequisites

### Required Services
1. **LM Studio** on port 1234
   - Load: `unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF`
   - Timeout: 300 seconds (thinking tokens need time)

2. **Ollama** on port 11434
   - Load: `qwen3-30b` 
   - Used for fast evidence binding only

### Test Service Status
```bash
# Test LM Studio (235B)
curl -X POST http://localhost:1234/v1/chat/completions \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Test"}]}'

# Test Ollama (30B)  
curl -X POST http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen3-30b","messages":[{"role":"user","content":"Test"}]}'
```

## Configuration

The hybrid setup is defined in `config.json`:

```json
{
  "llm": {
    "api_base": "http://localhost:1234/v1",
    "model": "unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF", 
    "timeout": 300
  },
  "binding_llm": {
    "api_base": "http://localhost:11434/v1",
    "model": "qwen3-30b",
    "timeout": 120
  },
  "binder": {
    "semantic": {
      "enabled": true,
      "model_name": "BAAI/bge-m3",
      "max_docs": 20000,
      "topk": 200,
      "weight_semantic": 0.65,
      "weight_bm25": 0.35,
      "batch_size": 128,
      "force_cache_reuse": false
    },
    "parquet": {
      "max_files": 200,
      "file_selection": "round_robin"
    }
  }
}
```

Enable `force_cache_reuse` to reuse a previously built FAISS index even when
the corpus fingerprint changes. The flag defaults to `false` so callers must
opt in explicitly if they want to skip rebuilding the semantic index.

## Core Components

### Pipeline Scripts
- **`ddl_paper_only_pipeline.py`** - Main orchestrator with hybrid routing
- **`semantic_ddl_pipeline.py`** - Semantic concept pairing system
- **`ddl_binder.py`** - Evidence binding (uses Ollama for speed)

### Processing Components  
- **`ddl_generator.py`** - Daydream generation (235B thinking)
- **`ddl_critic.py`** - Quality evaluation and filtering
- **`weave_composer_freeflow.py`** - Final editorial composition
- **`hybrid_retrieval_system.py`** - BGE-M3 + BM25 vector search

### Concept Extraction
- **`paper_concepts.py`** - LLM-based concept extraction from papers
- **`semantic_corpus_sampler.py`** - Intelligent corpus concept matching

## Data Source Clients

The project includes `paper_sources.py`, which provides helpers for pulling
research content from a variety of APIs:

- **OpenAlex** for canonical IDs, concepts and citation graphs
- **arXiv** preprints filtered for AI/ML categories
- **OpenReview** submissions and reviews from major conferences
- **Crossref** DOI metadata and references
- **Semantic Scholar** citation lookups
- **Alignment blogs** (Alignment Forum, LessWrong, major lab blogs) via
  sitemap scraping

OpenAlex is used as the spine for deduplication and to unify records from other
sources through shared identifiers such as DOI or arXiv IDs.

## Input Files

Located in `pipeline/input/`:
- **`CBH.txt`** - Critical Brain Hypothesis paper
- **`Autoalignment.txt`** - Auto-alignment safety research
- **`honest.txt`** - Honesty in AI systems
- **`backdoor.txt`** - Backdoor attack analysis
- Plus others for various AI safety topics

## V8 Enhancements

### Semantic Retrieval System
- **Hybrid Ranking**: BGE-M3 embeddings (65%) + BM25 lexical (35%) 
- **Performance**: 20-40% more bound quotes vs lexical-only
- **Auto-fallback**: Gracefully disables if dependencies missing
- **Configurable**: Tune weights, topk, max_docs via `config.json`

### Enhanced Acceptance Criteria  
- **Content-based**: Rows scanned (2M default) or unique docs (50K default)
- **Legacy support**: Original 600-file threshold still available
- **CLI options**: `--accept-by {rows,docs,files}` with custom thresholds
- **Real metrics**: Based on actual content analyzed, not arbitrary counts

### System Reliability
- **Preflight checks**: `python scripts/preflight.py` validates all dependencies
- **Structured logging**: JSON logs with run_id, stage, event tracking
- **Heartbeat monitoring**: Progress tracking every 30s during long operations
- **Error handling**: Clear actionable messages for common failure modes

## Expected Results

With the V8 hybrid system:
- **Speed**: ~22 minutes for 3-pair analysis (vs ~45+ minutes single model)
- **Quality**: Full 235B reasoning for creative tasks + enhanced semantic retrieval
- **Binding**: Fast evidence retrieval with 30B model, improved recall with semantic ranking
- **Output**: Complete editorial analysis (1,500-3,000 words) with higher evidence density

## Output Structure

```
pipeline/output/DDL_{Paper}_{Timestamp}/
├── ddl/
│   ├── concepts.jsonl          # Extracted paper concepts
│   ├── pairs.sampled.jsonl     # Generated concept pairs  
│   ├── daydreams.generated.jsonl # Raw hypotheses
│   ├── daydreams.accepted.jsonl  # Filtered daydreams
│   ├── daydreams.bound.jsonl     # Evidence-bound (target)
│   └── editorial_freeflow.md     # Final analysis
└── manifest.json               # Run metadata
```

## Monitoring & Observability

### Real-time Progress Tracking
```bash
# Monitor heartbeat events during long operations  
tail -f pipeline/output/*/binder.log | grep '"event":"heartbeat"'

# Track progress percentage
tail -f pipeline/output/*/binder.log | grep '"progress_pct"' | jq '.progress_pct'

# Monitor specific stages
tail -f pipeline/output/*/binder.log | grep '"stage":"semantic_indexing"'

# View error rates
tail -f pipeline/output/*/binder.log | grep '"nli_timeouts"' | jq '.nli_timeouts'
```

### Run Analytics
```bash
# Extract performance metrics for a run
grep '"run_id":"your-run-id"' binder.log | jq '{stage,event,progress_pct,timestamp}'

# Analyze corpus loading performance
grep '"stage":"corpus_loading"' binder.log | jq '{files_processed,rows_loaded}'

# Check acceptance metrics
cat pipeline/output/*/binder_metrics.json | jq '{rows_scanned,unique_docs_scanned,bound_quotes}'
```

## Troubleshooting

### Common Issues
1. **Dependencies**: Run `python3 scripts/preflight.py` to validate all requirements
2. **Timeout errors**: Increase timeout for 235B model (thinking tokens are slow)
3. **Connection refused**: Ensure both LM Studio (1234) and Ollama (11434) are running
4. **Acceptance failures**: Check `binder_metrics.json` for actual vs expected thresholds
5. **Memory issues**: 235B model + BGE-M3 require significant RAM

### Debug Commands
```bash
# Validate system before running
python3 scripts/preflight.py

# Check running processes
ps aux | grep -E "(lmstudio|ollama)"

# Test individual components
python3 -c "from ddl_binder import DDLEvidenceBinder; print('Binder OK')"
python3 -c "from ddl_generator import DDLGenerator; print('Generator OK')"

# Verify semantic retrieval
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3'); print('BGE-M3 OK')"
```

## Version History

- **V8**: Hybrid model system (235B + 30B) 
- **V7**: Single model with enhanced timeouts
- **V6**: Semantic pairing breakthrough
- **V5**: Hardened system with always-compose

---

*V8 represents the optimization of the DDL system - combining the creative power of large thinking models with the speed of efficient models for time-sensitive tasks.*
