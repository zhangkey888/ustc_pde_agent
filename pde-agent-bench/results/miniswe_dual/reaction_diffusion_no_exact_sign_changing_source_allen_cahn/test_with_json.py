import numpy as np
import json
import time
from solver import solve

# Load the actual case_spec
with open("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/agent_output/case_spec.json") as f:
    case_spec = json.load(f)

print("Keys:", list(case_spec.keys()))
print("oracle_config keys:", list(case_spec.get("oracle_config", {}).keys()))

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u = result['u']
print(f"Solve: {elapsed:.2f}s, shape={u.shape}, range=[{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
print(f"NaN count: {np.isnan(u).sum()}")
print(f"Solver info: {result['solver_info']}")
