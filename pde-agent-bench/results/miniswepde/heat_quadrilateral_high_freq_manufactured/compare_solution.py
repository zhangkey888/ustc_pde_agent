import numpy as np
from solver import solve

# Run solver
result = solve({
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
})

my_u = result['u']

# Load reference
ref_data = np.load('oracle_output/reference.npz')
ref_u = ref_data['u_star']
x_ref = ref_data['x']
y_ref = ref_data['y']

print(f"My solution shape: {my_u.shape}")
print(f"Reference shape: {ref_u.shape}")
print(f"Reference x shape: {x_ref.shape}, y shape: {y_ref.shape}")

# Check if grids match
nx, ny = 50, 50
x_expected = np.linspace(0.0, 1.0, nx)
y_expected = np.linspace(0.0, 1.0, ny)

if x_ref.shape == (nx,) and y_ref.shape == (ny,):
    print("Grid dimensions match")
    
    # Compute error
    error = np.abs(my_u - ref_u)
    max_error = np.max(error)
    mean_error = np.mean(error)
    l2_error = np.sqrt(np.mean(error**2))
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    print(f"L2 error: {l2_error:.2e}")
    
    # Check against requirement 9.19e-03
    if max_error <= 9.19e-03:
        print("PASS: Max error meets requirement")
    else:
        print(f"FAIL: Max error {max_error:.2e} > 9.19e-03")
else:
    print("Grid mismatch")
