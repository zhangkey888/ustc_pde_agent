import numpy as np
import json
import time
from solver import solve

with open("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/agent_output/case_spec.json") as f:
    case_spec = json.load(f)

# Standard solution (N=140, P1)
t0 = time.time()
r1 = solve(case_spec)
e1 = time.time() - t0
u1 = r1['u']
print(f"Standard (N=140,P1): {e1:.2f}s, range=[{np.nanmin(u1):.6f}, {np.nanmax(u1):.6f}]")

# Reference-quality solution (N=260, P2)
case_spec2 = json.loads(json.dumps(case_spec))
case_spec2["mesh_resolution"] = 260
case_spec2["element_degree"] = 2
t0 = time.time()
r2 = solve(case_spec2)
e2 = time.time() - t0
u2 = r2['u']
print(f"Reference (N=260,P2): {e2:.2f}s, range=[{np.nanmin(u2):.6f}, {np.nanmax(u2):.6f}]")

diff = u2 - u1
rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(u2**2))
print(f"rel_L2 error: {rel_l2:.6e}, max_diff: {np.max(np.abs(diff)):.6e}")
print(f"Threshold: 3.67e-02")
print(f"PASS: {rel_l2 < 3.67e-02}")
