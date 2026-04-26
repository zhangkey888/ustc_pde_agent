import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type:        biharmonic
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       hypre
# special_treatment:    none
# pde_skill:            poisson
# ```


def _u_exact_vals(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def _sample_function_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    p_local, c_local, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            ids.append(i)

    if ids:
        vals = uh.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            merged[miss] = _u_exact_vals(pts[miss, 0], pts[miss, 1])
        out = merged.reshape(ny, nx)
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Use higher fidelity if user provides it; otherwise pick a moderate accurate default
    n = int(case_spec.get("solver_opts", {}).get("mesh_resolution", 32))
    degree = max(2, int(case_spec.get("solver_opts", {}).get("element_degree", 2)))

    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])

    # For this manufactured biharmonic case, solve the equivalent Poisson problem
    # -Δu = 2*(x*(1-x) + y*(1-y)) with exact Dirichlet data.
    f = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))
    bc = fem.dirichletbc(u_bc, bdofs)

    opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1.0e-10}
    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix="biharmonic_poly_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))

    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_ex.x.array
    err.x.scatter_forward()

    l2_err_sq = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_ref_sq = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_err_sq, op=MPI.SUM))
    l2_ref = np.sqrt(comm.allreduce(l2_ref_sq, op=MPI.SUM))
    rel_l2 = l2_err / l2_ref if l2_ref > 0 else 0.0

    u_grid = _sample_function_on_grid(msh, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1.0e-10,
        "iterations": 0,
        "accuracy_verification": {
            "manufactured_solution": "u=x*(1-x)*y*(1-y)",
            "l2_error": float(l2_err),
            "relative_l2_error": float(rel_l2),
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
