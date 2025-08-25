#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def run_enhanced_4x_test():
    """Run enhanced V6 test with 4x compute budget (12-15 pairs)"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"enhanced_4x_test_{timestamp}.log"
    
    print(f"🚀 Enhanced V6 Test - 4x Compute Budget")
    print(f"📝 Log file: {log_file}")
    print("=" * 60)
    print("🎯 ENHANCEMENTS IMPLEMENTED:")
    print("   • k=20 retrieval with per-doc cap=2")  
    print("   • 3 query variants per concept")
    print("   • Enhanced concept filtering")
    print("   • Windowed evidence snippets")
    print("   • Progress metrics tracking")
    print()
    print("🎯 SUCCESS TARGETS:")
    print("   • 12-15 high-quality pairs")
    print("   • ≥8 pairs from 3-5 distinct sources") 
    print("   • ≥2 bound daydreams with cross-doc citations")
    print("=" * 60)
    
    # 4x compute test: 12 pairs target
    cmd = [
        "python3", "ddl_paper_only_pipeline.py", 
        "pipeline/input/honest.txt", 
        "--config", "config.json",
        "--pairs", "12"  # 4x increase from 3
    ]
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== Enhanced V6 Test - 4x Compute - {timestamp} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("Enhancements: k=20, 3 queries/concept, per-doc cap=2, enhanced filtering\n")
            f.write("Target: 12-15 pairs from 3-5 distinct sources\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream with enhanced monitoring
            for line in process.stdout:
                line = line.rstrip()
                
                # Highlight key progress indicators
                if "🔧 CONFIG:" in line:
                    print(f"⚙️  {line}")
                elif "📊 METRICS:" in line:
                    print(f"📊 {line}")
                elif "🏆 Rank-based:" in line:
                    print(f"🎯 {line}")
                elif "Enhanced search:" in line:
                    print(f"🔍 {line}")
                elif "Query" in line and "/" in line:
                    print(f"📝 {line}")
                elif "Generated" in line and "semantic concept pairs" in line:
                    print(f"🎉 {line}")
                elif "bound=" in line and not "0" in line.split("bound=")[1].split()[0]:
                    print(f"🏆 BREAKTHROUGH: {line}")
                else:
                    print(line)
                
                f.write(line + "\n")
                f.flush()
            
            process.wait()
            
            print(f"\n📋 Enhanced test complete. Check {log_file} for full results.")
            return log_file
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    run_enhanced_4x_test()