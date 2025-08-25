#!/usr/bin/env python3
"""
DDL V8 Preflight Checks

Validates system dependencies and service availability before pipeline execution.
Returns 0 on success, non-zero on failure with actionable error messages.
"""

import sys
import subprocess
import json
from pathlib import Path

def check_parquet_engine():
    """Verify parquet engine availability and functionality."""
    print("🔍 Checking parquet engine...")
    
    # First check if libraries are importable
    engine_name = None
    engine_version = None
    
    try:
        import pyarrow
        engine_name = "pyarrow"
        engine_version = pyarrow.__version__
        print(f"✅ PyArrow {engine_version} found")
    except ImportError:
        try:
            import fastparquet
            engine_name = "fastparquet"
            engine_version = fastparquet.__version__
            print(f"✅ FastParquet {engine_version} found")
        except ImportError:
            print("❌ No parquet engine found")
            print("   → Install: pip install pyarrow>=15.0.0  (recommended)")
            print("   → Alternative: pip install fastparquet>=2024.5.0")
            return False
    
    # Test actual parquet functionality
    print("🔍 Testing parquet read/write functionality...")
    try:
        import pandas as pd
        import tempfile
        import os
        
        # Create test data
        test_data = pd.DataFrame({
            'doc_id': ['test_doc_1', 'test_doc_2'],
            'text': ['Sample text content', 'Another test document'],
            'author': ['Test Author', 'Another Author'],
            'year': [2023, 2024]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            try:
                # Write parquet file
                test_data.to_parquet(tmp.name, engine=engine_name)
                
                # Read it back
                read_back = pd.read_parquet(tmp.name, engine=engine_name)
                
                # Verify round trip worked
                if len(read_back) != 2 or set(read_back.columns) != {'doc_id', 'text', 'author', 'year'}:
                    raise ValueError("Parquet round-trip verification failed")
                    
                print(f"✅ Parquet engine {engine_name} functional test passed")
                return True
                
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                    
    except Exception as e:
        print(f"❌ Parquet engine functional test failed: {e}")
        print(f"   Engine: {engine_name} {engine_version}")
        print("   This suggests a corrupted installation.")
        print(f"   → Try: pip uninstall {engine_name}")
        print(f"   → Then: pip install {engine_name}>={'15.0.0' if engine_name == 'pyarrow' else '2024.5.0'}")
        return False

def check_embedding_model():
    """Verify BGE-M3 embedding model availability."""
    print("🔍 Checking embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-m3')
        print(f"✅ BGE-M3 model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ BGE-M3 model failed to load: {e}")
        print("   → Install: pip install sentence-transformers>=2.7")
        print("   → First run downloads ~2GB model automatically")
        return False

def check_llm_endpoints():
    """Verify LM Studio and Ollama API endpoints."""
    print("🔍 Checking LLM endpoints...")
    
    endpoints = [
        ("LM Studio (Qwen 235B)", "http://localhost:1234/v1/chat/completions"),
        ("Ollama (Qwen 30B)", "http://localhost:11434/v1/chat/completions")
    ]
    
    success = True
    for name, url in endpoints:
        try:
            import requests
            # Simple test payload
            payload = {
                "model": "qwen",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            response = requests.post(
                url, 
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ {name} responding")
            else:
                print(f"⚠️  {name} returned HTTP {response.status_code}")
                print(f"   → Check service status on {url.split('/v1')[0]}")
                success = False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {name} connection refused")
            port = url.split(':')[2].split('/')[0]
            print(f"   → Start service on port {port}")
            success = False
        except requests.exceptions.Timeout:
            print(f"⚠️  {name} timeout (service may be starting)")
        except Exception as e:
            print(f"❌ {name} error: {e}")
            success = False
    
    return success

def check_config_file():
    """Verify config.json exists and is valid."""
    print("🔍 Checking configuration...")
    
    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ config.json not found")
        print("   → Create config.json with LLM endpoints")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = ['llm', 'corpus_path']
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            print(f"❌ config.json missing keys: {missing}")
            return False
            
        print("✅ config.json valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ config.json invalid JSON: {e}")
        return False

def check_corpus_path():
    """Verify corpus directory exists.""" 
    print("🔍 Checking corpus access...")
    
    try:
        with open("config.json") as f:
            config = json.load(f)
        
        corpus_path = Path(config['corpus_path'])
        
        if corpus_path.is_file() and corpus_path.suffix == '.parquet':
            print(f"✅ Corpus file: {corpus_path}")
            return True
        elif corpus_path.is_dir():
            parquet_files = list(corpus_path.glob("*.parquet"))
            if parquet_files:
                print(f"✅ Corpus directory: {len(parquet_files)} parquet files")
                return True
            else:
                print(f"❌ No parquet files in {corpus_path}")
                return False
        else:
            print(f"❌ Corpus path not found: {corpus_path}")
            return False
            
    except Exception as e:
        print(f"❌ Corpus check failed: {e}")
        return False

def main():
    """Run all preflight checks."""
    print("🚀 DDL V8 Preflight Checks\n")
    
    checks = [
        check_parquet_engine,
        check_embedding_model, 
        check_config_file,
        check_corpus_path,
        check_llm_endpoints
    ]
    
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            results.append(False)
        print()  # spacing
    
    if all(results):
        print("🎉 All preflight checks passed! System ready.")
        return 0
    else:
        failed = len([r for r in results if not r])
        print(f"⚠️  {failed} preflight check(s) failed. Fix issues above before running pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())