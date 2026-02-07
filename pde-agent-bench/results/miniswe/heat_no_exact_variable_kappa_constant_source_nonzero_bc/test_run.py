import subprocess
import sys

result = subprocess.run([sys.executable, 'solver.py', '--resolution', '8', '--degree', '1', '--outdir', 'test_output_debug', '--dt', '0.02'], 
                       capture_output=True, text=True)
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
print("Stderr:", result.stderr)
print("Files in output dir:")
import os
if os.path.exists('test_output_debug'):
    print(os.listdir('test_output_debug'))
