# DDL V8 - Hybrid Model System

## Overview
Version 8 implements a hybrid model approach for optimal performance:
- **Qwen 235B (LM Studio)**: Creative tasks (concept extraction, daydream generation, composition)  
- **Qwen 30B (Ollama)**: Evidence binding (3-5x faster than 235B)

## Architecture

### Model Distribution
- **Stage 1** (Paper Concepts): Qwen 235B-A22B-Thinking → LM Studio:1234
- **Stage 2.2** (Daydream Generation): Qwen 235B-A22B-Thinking → LM Studio:1234  
- **Stage 2.3** (Critic Filtering): Qwen 235B-A22B-Thinking → LM Studio:1234
- **Stage 3** (Evidence Binding): Qwen 30B → Ollama:11434 ⚡
- **Stage 4** (Final Composition): Qwen 235B-A22B-Thinking → LM Studio:1234

### Configuration
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
  }
}
```

## Performance Benefits
- **Speed**: Evidence binding ~3-5x faster with 30B model
- **Quality**: Maintains 235B thinking for creative analysis
- **Parallel**: LM Studio + Ollama can run simultaneously
- **Memory**: More efficient resource utilization

## Usage
```bash
cd /Users/willkirby/scrape\ 2/LW_scrape/v8
python3 ddl_paper_only_pipeline.py pipeline/input/CBH.txt --pairs 200
```

## Prerequisites
1. LM Studio running Qwen 235B-A22B-Thinking on port 1234
2. Ollama running qwen3-30b on port 11434
3. Shared corpus access (v7 corpus path maintained)

## Key Changes from V7
- Added `binding_llm` configuration section
- Modified `ddl_binder.py` to default to Ollama
- Updated `_binding_worker_with_queue` to use hybrid config
- Maintained all v7 scripts and input files
- Clean output directory for fresh runs