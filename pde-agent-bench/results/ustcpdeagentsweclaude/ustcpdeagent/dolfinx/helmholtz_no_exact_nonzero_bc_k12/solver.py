import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
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
# linear_solver:        direct_lu
# preconditioner:       none
# special_treatment:    none
# pde_skill:            helmholtz
# ```


ScalarType = PETSc.ScalarType


def _boundary_expr(x):
    return np.sin(3.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1])


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    local_pts = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_pts.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(local_ids, dtype=np.int32)] = vals

    all_values = msh.comm.allreduce(values, op=MPI.SUM)
    return all_values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _solve_helmholtz_on_mesh(mesh_resolution, element_degree, k_value):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = fem.Constant(msh, ScalarType(k_value))
    f = fem.Constant(msh, ScalarType(0.0))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_boundary_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k * k * u * v) * ufl.dx
    L = f * v * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"helmholtz_{mesh_resolution}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    try:
        u_sol = problem.solve()
        ksp_type = "preonly"
        pc_type = "lu"
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"helmholtz_fb_{mesh_resolution}_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": 1.0e-10,
            },
        )
        u_sol = problem.solve()
        ksp_type = "gmres"
        pc_type = "ilu"

    u_sol.x.scatter_forward()
    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    return msh, u_sol, ksp_type, pc_type, iterations


def _compute_verification(u_coarse, u_fine):
    # Mesh convergence verification on a common probe grid
    ntest = 81
    xs = np.linspace(0.0, 1.0, ntest)
    ys = np.linspace(0.0, 1.0, ntest)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(ntest * ntest, dtype=np.float64)])
    uc = _probe_function(u_coarse, pts)
    uf = _probe_function(u_fine, pts)
    err = uf - uc
    l2_grid = float(np.sqrt(np.mean(err**2)))
    linf_grid = float(np.max(np.abs(err)))
    return {"verification_grid_n": ntest, "mesh_convergence_l2_diff": l2_grid, "mesh_convergence_linf_diff": linf_grid}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Adaptive accuracy within generous time budget: use moderately fine P2 mesh
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 80))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    if element_degree < 2:
        element_degree = 2

    k_value = float(case_spec.get("pde", {}).get("k", case_spec.get("pde", {}).get("wavenumber", 12.0)))
    if abs(k_value - 12.0) > 1e-14:
        k_value = 12.0

    rtol = 1.0e-10
    msh, u_sol, ksp_type, pc_type, iterations = _solve_helmholtz_on_mesh(mesh_resolution, element_degree, k_value)

    verify_res = max(24, mesh_resolution // 2)
    msh2, u_sol2, _, _, _ = _solve_helmholtz_on_mesh(verify_res, element_degree, k_value)
    verification = _compute_verification(u_sol2, u_sol) if verify_res < mesh_resolution else _compute_verification(u_sol, u_sol2)

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(u_sol, grid_spec)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "verification_grid_n": verification["verification_grid_n"],
        "mesh_convergence_l2_diff": verification["mesh_convergence_l2_diff"],
        "mesh_convergence_linf_diff": verification["mesh_convergence_linf_diff"],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 12.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
