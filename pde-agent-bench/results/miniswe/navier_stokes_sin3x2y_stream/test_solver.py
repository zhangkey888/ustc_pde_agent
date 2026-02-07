import subprocess
import sys
import os
os.makedirs("test_out", exist_ok=True)
subprocess.run([sys.executable, "solver.py", "--resolution", "16", "--degree", "1", "--outdir", "test_out"])
