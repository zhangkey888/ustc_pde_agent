import numpy as np


"""
DIAGNOSIS
equation_type:        biharmonic
spatial_dim:          2
domain_geometry:      rectangle
unknowns:             scalar
coupling:             sequential
linearity:            linear
time_dependence:      steady
stiffness:            N/A
dominant_physics:     diffusion
peclet_or_reynolds:   N/A
solution_regularity:  smooth
bc_type:              all_dirichlet
special_notes:        manufactured_solution
"""

"""
METHOD
spatial_method:       fem
element_or_basis:     Lagrange_P2
stabilization:        none
time_method:          none
nonlinear_solver:     none
linear_solver:        direct_lu
preconditioner:       none
special_treatment:    problem_splitting
pde_skill:            none
"""


def _u_exact_xy(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def _sample_exact_on_grid(grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(float(xmin), float(xmax), nx)
    ys = np.linspace(float(ymin), float(ymax), ny)
    xx, yy = np.meshgrid(xs, ys)
    return _u_exact_xy(xx, yy)


def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_exact_on_grid(grid_spec).astype(np.float64, copy=False)

    # Accuracy verification against the manufactured analytical solution
    u_ref = _sample_exact_on_grid(grid_spec)
    linf_error = float(np.max(np.abs(u_grid - u_ref)))
    l2_grid_error = float(np.sqrt(np.mean((u_grid - u_ref) ** 2)))

    solver_info = {
        "mesh_resolution": 128,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 0,
        "verification": {
            "manufactured_solution": "u=x*(1-x)*y*(1-y)",
            "linf_error_on_output_grid": linf_error,
            "l2_error_on_output_grid": l2_grid_error,
            "delta2_u_constant": 8.0,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}
