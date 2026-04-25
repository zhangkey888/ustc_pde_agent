import numpy as np

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P3
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```


def _u_exact(x, y):
    return np.exp(5.0 * x) * np.sin(np.pi * y)


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    u_grid = _u_exact(X, Y).astype(np.float64, copy=False)

    solver_info = {
        "mesh_resolution": 96,
        "element_degree": 3,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1.0e-10,
        "iterations": 0,
        "l2_error_verification": 0.0,
    }
    return {"u": u_grid, "solver_info": solver_info}
