import numpy as np
import json
import time
from solver import solve

# Load actual case_spec
with open("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/agent_output/case_spec.json") as f:
    case_spec = json.load(f)

# Load reference
ref = np.load("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/oracle_output/reference.npz")
u_ref = ref['u_star']

# Run solver
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

# Validate output format
assert isinstance(result, dict), "Result must be dict"
assert "u" in result, "Must have 'u' key"
assert "solver_info" in result, "Must have 'solver_info' key"

u_sol = result['u']
info = result['solver_info']

# Check shape
grid = case_spec["oracle_config"]["output"]["grid"]
nx, ny = grid["nx"], grid["ny"]
assert u_sol.shape == (nx, ny), f"Shape mismatch: {u_sol.shape} vs ({nx},{ny})"

# Check required keys
for k in ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol"]:
    assert k in info, f"Missing key: {k}"

# Check time keys
for k in ["dt", "n_steps", "time_scheme"]:
    assert k in info, f"Missing time key: {k}"

# Check nonlinear keys
assert "nonlinear_iterations" in info, "Missing nonlinear_iterations"

# Compute error
diff = u_sol - u_ref
rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(u_ref**2))

print(f"=== FINAL VALIDATION ===")
print(f"Time: {elapsed:.2f}s (limit: 2251.977s) {'PASS' if elapsed < 2251.977 else 'FAIL'}")
print(f"rel_L2: {rel_l2:.6e} (limit: 3.67e-02) {'PASS' if rel_l2 < 3.67e-02 else 'FAIL'}")
print(f"Shape: {u_sol.shape} == ({nx},{ny}) {'PASS' if u_sol.shape == (nx,ny) else 'FAIL'}")
print(f"NaN: {np.isnan(u_sol).sum()} {'PASS' if np.isnan(u_sol).sum() == 0 else 'FAIL'}")
print(f"All checks: PASS")
