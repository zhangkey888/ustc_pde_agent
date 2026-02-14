import numpy as np
import sys
import time
sys.path.insert(0, '.')
from solver import solve

def test_case_1():
    """Basic Poisson with homogeneous Dirichlet BC"""
    return {
        'pde': {
            'type': 'poisson',
            'coefficients': {'kappa': 1.0},
            'source': {'f': 1.0},
            'boundary_conditions': {
                'dirichlet': {'value': 0.0}
            }
        },
        'domain': {
            'bounds': [[0.0, 0.0], [1.0, 1.0]]
        }
    }

def test_case_2():
    """Poisson with non-homogeneous Dirichlet BC"""
    return {
        'pde': {
            'type': 'poisson',
            'coefficients': {'kappa': 2.0},
            'source': {'f': 0.5},
            'boundary_conditions': {
                'dirichlet': {'value': 1.0}
            }
        },
        'domain': {
            'bounds': [[0.0, 0.0], [1.0, 1.0]]
        }
    }

def test_case_3():
    """Poisson with different kappa"""
    return {
        'pde': {
            'type': 'poisson',
            'coefficients': {'kappa': 0.1},
            'source': {'f': 2.0},
            'boundary_conditions': {
                'dirichlet': {'value': 0.0}
            }
        },
        'domain': {
            'bounds': [[0.0, 0.0], [1.0, 1.0]]
        }
    }

def test_case_4():
    """Poisson with time field (should be ignored for elliptic)"""
    return {
        'pde': {
            'type': 'poisson',
            'coefficients': {'kappa': 1.0},
            'source': {'f': 1.0},
            'boundary_conditions': {
                'dirichlet': {'value': 0.0}
            },
            'time': {'t_end': 1.0, 'dt': 0.1}  # Should be ignored
        },
        'domain': {
            'bounds': [[0.0, 0.0], [1.0, 1.0]]
        }
    }

print("Running comprehensive tests...")
test_cases = [test_case_1(), test_case_2(), test_case_3(), test_case_4()]
case_names = ["Homogeneous BC", "Non-homogeneous BC", "Different kappa", "With time field"]

for i, (case, name) in enumerate(zip(test_cases, case_names), 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = solve(case)
    elapsed = time.time() - start_time
    
    u_grid = result['u']
    solver_info = result['solver_info']
    
    print(f"Time: {elapsed:.3f} seconds")
    print(f"Mesh resolution: {solver_info['mesh_resolution']}")
    print(f"Solver: {solver_info['ksp_type']} with {solver_info['pc_type']}")
    print(f"Iterations: {solver_info['iterations']}")
    print(f"Solution range: [{u_grid.min():.6f}, {u_grid.max():.6f}]")
    
    # Check time constraint
    if elapsed > 9.190:
        print(f"⚠ WARNING: Time {elapsed:.3f}s exceeds constraint 9.190s")
    else:
        print(f"✓ Time constraint satisfied")
    
    # Basic sanity checks
    if not np.any(np.isnan(u_grid)):
        print("✓ No NaN values in solution")
    else:
        print("✗ NaN values found in solution!")
    
    if np.abs(u_grid).max() < 1e6:  # Reasonable magnitude
        print("✓ Solution magnitude is reasonable")
    else:
        print("✗ Solution magnitude seems too large!")

print(f"\n{'='*60}")
print("All tests completed!")
print(f"{'='*60}")
