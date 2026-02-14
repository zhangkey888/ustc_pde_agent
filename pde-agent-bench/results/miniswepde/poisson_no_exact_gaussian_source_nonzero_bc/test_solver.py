import time
import solver
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0},
        "source": "gaussian"
    }
}
start = time.time()
result = solver.solve(case_spec)
end = time.time()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Max solution value: {result['u'].max()}")
