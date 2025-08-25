#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def run_v6_with_logging():
    """Run v6 DDL pipeline with comprehensive logging"""
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"v6_ddl_test_{timestamp}.log"
    
    print(f"🚀 Starting v6 DDL Pipeline Test...")
    print(f"📝 Log file: {log_file}")
    print(f"🔍 Monitor with: tail -f {log_file}")
    print("=" * 60)
    
    # Run the pipeline with logging and config
    cmd = [
        "python3", "ddl_paper_only_pipeline.py", 
        "pipeline/input/honest.txt", 
        "--config", "config.json",  # 🔧 CRITICAL: Load config.json
        "--pairs", "3"
    ]
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== v6 DDL Pipeline Test - {timestamp} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            # Run with real-time logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line.rstrip())  # Print to console
                f.write(line)  # Write to log
                f.flush()  # Ensure immediate write
            
            process.wait()
            
            f.write(f"\n=== Process completed with exit code: {process.returncode} ===\n")
            
        print(f"\n📝 Complete log saved to: {log_file}")
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

if __name__ == "__main__":
    success = run_v6_with_logging()
    if success:
        print("✅ v6 pipeline completed successfully!")
    else:
        print("❌ v6 pipeline failed - check the log for details")