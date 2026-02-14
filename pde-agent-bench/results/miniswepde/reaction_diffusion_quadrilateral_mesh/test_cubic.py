import numpy as np
import solver

print("Testing cubic nonlinear reaction case")
case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "time": {
            "t_end": 0.4,
            "dt": 0.01,
            "scheme": "backward_euler"
        },
        "reaction": {
            "type": "cubic",
            "alpha": 0.1
        }
    }
}

result = solver.solve(case_spec)
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Has nonlinear_iterations: {'nonlinear_iterations' in result['solver_info']}")
if 'nonlinear_iterations' in result['solver_info']:
    print(f"Nonlinear iterations per step: {result['solver_info']['nonlinear_iterations'][:5]}...")
print("Cubic reaction test completed")
