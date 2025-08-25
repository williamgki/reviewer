#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def quick_corpus_test():
    """Quick test focused on corpus path fix - just 3 pairs, minimal time"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"quick_corpus_test_{timestamp}.log"
    
    print(f"🚀 Quick Corpus Path Test")
    print(f"📝 Log file: {log_file}")
    print("=" * 50)
    print("🎯 TESTING: Evidence binding corpus path fix")
    print("   • Target: 3 pairs (minimal)")
    print("   • Focus: Stage 3 evidence binding success")
    print("   • Expected: bound_daydreams > 0")
    print("=" * 50)
    
    # Minimal test: just 3 pairs
    cmd = [
        "python3", "ddl_paper_only_pipeline.py", 
        "pipeline/input/honest.txt", 
        "--config", "config.json",
        "--pairs", "3"  # Minimal for speed
    ]
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== Quick Corpus Path Test - {timestamp} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("Purpose: Test corpus path fix for evidence binding\n")
            f.write("Target: bound_daydreams > 0 in Stage 3\n")
            f.write("=" * 50 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream with targeted monitoring
            for line in process.stdout:
                line = line.rstrip()
                
                # Highlight key progress indicators
                if "Loading chunks from" in line:
                    print(f"✅ CORPUS LOADING: {line}")
                elif "Error loading corpus" in line:
                    print(f"❌ CORPUS ERROR: {line}")
                elif "Stage 3:" in line:
                    print(f"🎯 EVIDENCE BINDING: {line}")
                elif "bound=" in line and "0" not in line.split("bound=")[1].split()[0]:
                    print(f"🏆 SUCCESS: {line}")
                elif "bound=0" in line:
                    print(f"❌ STILL FAILING: {line}")
                elif "Generated" in line and "pairs" in line:
                    print(f"📊 {line}")
                elif "word count=" in line:
                    print(f"📝 {line}")
                else:
                    # Show abbreviated output for speed
                    if any(x in line for x in ["✓", "Stage", "bound", "ERROR", "Loading"]):
                        print(line)
                
                f.write(line + "\n")
                f.flush()
            
            process.wait()
            
            print(f"\n📋 Quick test complete. Check {log_file} for details.")
            return log_file
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    quick_corpus_test()