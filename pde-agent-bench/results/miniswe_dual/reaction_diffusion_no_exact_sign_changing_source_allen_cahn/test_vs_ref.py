import numpy as np
import json
import time
from solver import solve

# Load reference
ref = np.load("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/oracle_output/reference.npz")
u_ref = ref['u_star']
print(f"Reference: shape={u_ref.shape}, range=[{np.min(u_ref):.6f}, {np.max(u_ref):.6f}]")

# Load case_spec
with open("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/agent_output/case_spec.json") as f:
    case_spec = json.load(f)

# Run solver
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0
u_sol = result['u']

print(f"Solver: {elapsed:.2f}s, shape={u_sol.shape}")
print(f"Solver range: [{np.nanmin(u_sol):.6f}, {np.nanmax(u_sol):.6f}]")

# Compute rel_L2_grid error
diff = u_sol - u_ref
l2_err = np.sqrt(np.mean(diff**2))
l2_ref = np.sqrt(np.mean(u_ref**2))
rel_l2 = l2_err / l2_ref if l2_ref > 0 else l2_err
print(f"\nrel_L2_grid error: {rel_l2:.6e}")
print(f"Threshold: 3.67e-02")
print(f"PASS: {rel_l2 < 3.67e-02}")
print(f"Max absolute diff: {np.max(np.abs(diff)):.6e}")
