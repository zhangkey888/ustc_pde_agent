import numpy as np
from solver import solve

# Create a minimal case_spec for heat equation
case_spec = {
    'pde': {
        'type': 'heat',
        'time': {
            't_end': 0.1,
            'dt': 0.01,
            'scheme': 'backward_euler'
        },
        'coefficients': {
            'kappa': 1.0
        },
        'manufactured_solution': 'exp(-t)*sin(pi*x)*sin(pi*y)'
    },
    'domain': {
        'type': 'unit_square',
        'bounds': [[0,1], [0,1]]
    }
}

try:
    result = solve(case_spec)
    print("Success! Keys in result:", result.keys())
    if 'u' in result:
        print("u shape:", result['u'].shape)
    if 'solver_info' in result:
        print("solver_info:", result['solver_info'])
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
