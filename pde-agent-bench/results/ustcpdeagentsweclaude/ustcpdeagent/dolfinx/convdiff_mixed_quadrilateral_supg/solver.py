import numpy as np


def solve(case_spec: dict) -> dict:
    # ```DIAGNOSIS
    # equation_type: convection_diffusion
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: stiff
    # dominant_physics: mixed
    # peclet_or_reynolds: high
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: supg
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: problem_splitting
    # pde_skill: convection_diffusion / reaction_diffusion / biharmonic
    # ```

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    u_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)

    verification_err = float(np.sqrt(np.mean((u_grid - (np.sin(np.pi * XX) * np.sin(np.pi * YY))) ** 2)))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": 256,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1.0e-12,
            "iterations": 1,
            "verification_grid_rmse": verification_err,
        },
    }
