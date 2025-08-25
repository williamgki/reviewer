#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def test_config_fixes():
    """Test all config propagation fixes with comprehensive logging"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"config_fixes_test_{timestamp}.log"
    
    print(f"🧪 Testing Config Propagation Fixes")
    print(f"📝 Log file: {log_file}")
    print("🔍 Key diagnostics to watch for:")
    print("   1. 🔧 CONFIG: min_semantic_score=0.01 (not 0.1)")
    print("   2. 🔍 Searching with k=10 (not 2)")
    print("   3. 🏆 Rank-based: Kept top N concepts")
    print("   4. ✅ Authors/sources not 'Unknown'")
    print("=" * 60)
    
    # Test with 5 pairs to make it quick but sufficient
    cmd = [
        "python3", "ddl_paper_only_pipeline.py", 
        "pipeline/input/honest.txt", 
        "--config", "config.json",  # Critical: load config.json
        "--pairs", "5"
    ]
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== Config Fixes Test - {timestamp} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("Expected diagnostics:\n")
            f.write("  🔧 SemanticDDL CONFIG: min_semantic_score=0.01\n") 
            f.write("  🔧 Semantic corpus concept sampler: min_semantic_score=0.01\n")
            f.write("  🔍 Searching with k=10\n")
            f.write("  🏆 Rank-based: Kept top N concepts\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream with highlighting of key diagnostics
            for line in process.stdout:
                line = line.rstrip()
                
                # Highlight key diagnostic lines
                if "🔧 CONFIG:" in line:
                    print(f"🎯 {line}")  # Config diagnostic
                elif "🔍 Searching with k=" in line:
                    print(f"🔍 {line}")  # K value diagnostic  
                elif "🏆 Rank-based:" in line:
                    print(f"🏆 {line}")  # Rank-based filtering
                elif "Generated" in line and "semantic concept pairs" in line:
                    print(f"🎉 {line}")  # Success indicator
                elif "bound=" in line and not "0" in line.split("bound=")[1].split()[0]:
                    print(f"🏆 BREAKTHROUGH: {line}")  # Ultimate success
                else:
                    print(line)
                
                f.write(line + "\n")
                f.flush()
            
            process.wait()
            
            print(f"\n📋 Test complete. Check {log_file} for full diagnostics.")
            
            # Quick analysis
            with open(log_file, 'r') as f:
                content = f.read()
                
            print("\n🔬 Quick Analysis:")
            
            if "min_semantic_score=0.01" in content:
                print("  ✅ Config propagation: WORKING")
            else:
                print("  ❌ Config propagation: FAILED")
                
            if "🔍 Searching with k=10" in content:
                print("  ✅ K value diagnostic: WORKING")  
            else:
                print("  ❌ K value diagnostic: FAILED")
                
            if "🏆 Rank-based:" in content:
                print("  ✅ Rank-based filtering: WORKING")
            else:
                print("  ❌ Rank-based filtering: FAILED")
                
            if "Generated" in content and "semantic concept pairs" in content:
                pairs_line = [l for l in content.split('\n') if "Generated" in l and "semantic concept pairs" in l]
                if pairs_line and not "0 semantic concept pairs" in pairs_line[0]:
                    print("  ✅ Pair generation: WORKING")
                else:
                    print("  ❌ Pair generation: STILL FAILING")
            else:
                print("  ❌ Pair generation: NO DATA")
                
            return log_file
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_config_fixes()