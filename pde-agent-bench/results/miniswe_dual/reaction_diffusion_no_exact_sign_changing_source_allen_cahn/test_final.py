import numpy as np
import json
import time
from solver import solve

# Test 1: With actual case_spec.json
print("=== Test 1: Actual case_spec.json ===")
with open("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/agent_output/case_spec.json") as f:
    case_spec = json.load(f)

ref = np.load("../../miniswepde/reaction_diffusion_no_exact_sign_changing_source_allen_cahn/oracle_output/reference.npz")
u_ref = ref['u_star']

t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0
u_sol = result['u']

diff = u_sol - u_ref
rel_l2 = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(u_ref**2))
print(f"Time: {elapsed:.2f}s, rel_L2: {rel_l2:.6e}, PASS: {rel_l2 < 3.67e-02}")

# Test 2: With minimal pde spec
print("\n=== Test 2: Minimal pde spec ===")
case_spec2 = {
    "pde": {
        "type": "reaction_diffusion",
        "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
        "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
        "pde_params": {"epsilon": 0.05, "reaction": {"type": "allen_cahn", "lambda": 2.0}},
        "time": {"t_end": 0.2, "dt": 0.005, "scheme": "backward_euler"},
    },
}

t0 = time.time()
result2 = solve(case_spec2)
elapsed2 = time.time() - t0
u_sol2 = result2['u']

diff2 = u_sol2 - u_ref
rel_l2_2 = np.sqrt(np.mean(diff2**2)) / np.sqrt(np.mean(u_ref**2))
print(f"Time: {elapsed2:.2f}s, rel_L2: {rel_l2_2:.6e}, PASS: {rel_l2_2 < 3.67e-02}")

# Verify solver_info has all required keys
info = result['solver_info']
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol"]
time_keys = ["dt", "n_steps", "time_scheme"]
nonlinear_keys = ["nonlinear_iterations"]
print(f"\nRequired keys present: {all(k in info for k in required)}")
print(f"Time keys present: {all(k in info for k in time_keys)}")
print(f"Nonlinear keys present: {all(k in info for k in nonlinear_keys)}")
print(f"Has u_initial: {'u_initial' in result}")
