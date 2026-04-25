import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _u_exact_numpy_xy(x, y):
    return np.sin(3.0 * np.pi * (x + y)) * np.sin(np.pi * (x - y))


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = COMM.gather(values, root=0)
    if COMM.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & (~np.isnan(arr))
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at all requested output points.")
        out = out.reshape(ny, nx)
    else:
        out = None
    out = COMM.bcast(out, root=0)
    return out


def _solve_poisson(mesh_resolution=40, element_degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    domain = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(3.0 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    kappa = fem.Constant(domain, ScalarType(1.0))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(3.0 * np.pi * (X[0] + X[1])) * np.sin(np.pi * (X[0] - X[1])))
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_bench_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_H1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx))
    err_L2 = math.sqrt(COMM.allreduce(err_L2_local, op=MPI.SUM))
    err_H1_semi = math.sqrt(COMM.allreduce(err_H1_local, op=MPI.SUM))

    ksp = problem.solver
    return {
        "domain": domain,
        "uh": uh,
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "l2_error_verification": float(err_L2),
        "h1_semi_error_verification": float(err_H1_semi),
    }


def solve(case_spec: dict) -> dict:
    nx_out = int(case_spec["output"]["grid"]["nx"])
    ny_out = int(case_spec["output"]["grid"]["ny"])

    # Fast accurate default; modest upward refinement for larger output grids
    if max(nx_out, ny_out) >= 200:
        mesh_resolution, element_degree = 80, 3
    else:
        mesh_resolution, element_degree = 72, 3

    try:
        result = _solve_poisson(mesh_resolution=mesh_resolution, element_degree=element_degree,
                                ksp_type="cg", pc_type="hypre", rtol=1e-10)
    except Exception:
        result = _solve_poisson(mesh_resolution=mesh_resolution, element_degree=element_degree,
                                ksp_type="preonly", pc_type="lu", rtol=1e-12)

    u_grid = _sample_on_grid(result["domain"], result["uh"], case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": result["mesh_resolution"],
            "element_degree": result["element_degree"],
            "ksp_type": result["ksp_type"],
            "pc_type": result["pc_type"],
            "rtol": result["rtol"],
            "iterations": result["iterations"],
            "l2_error_verification": result["l2_error_verification"],
            "h1_semi_error_verification": result["h1_semi_error_verification"],
        },
    }
