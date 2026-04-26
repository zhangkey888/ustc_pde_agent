import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

DIAGNOSIS = """```DIAGNOSIS
equation_type: poisson
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
```"""

METHOD = """```METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: poisson
```"""

def _expr_to_function(V, expr):
    fn = fem.Function(V)
    fn.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return fn

def _sample_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.zeros((nx * ny, 3), dtype=np.float64)
    points[:, 0] = X.ravel()
    points[:, 1] = Y.ravel()
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)
    values = np.full((nx * ny,), np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)
    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_local), -1)[:, 0]
        values[np.array(ids_local, dtype=np.int32)] = vals
    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = np.full((nx * ny,), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        return merged.reshape((ny, nx))
    return None

def _manufactured_solution_check(comm):
    n = 24
    degree = 2
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = 2.0 * ufl.pi**2 * u_exact_expr
    f = _expr_to_function(V, f_expr)
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mms_", petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10})
    uh = problem.solve()
    uh.x.scatter_forward()
    u_exact = _expr_to_function(V, u_exact_expr)
    err_local = fem.assemble_scalar(fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx))
    err_l2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    return {"mms_l2_error": float(err_l2)}

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]
    mesh_resolution = 128
    element_degree = 3
    rtol = 1e-10
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    rhs_expr = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])
    f = _expr_to_function(V, rhs_expr)
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_", petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol, "ksp_atol": 1e-14, "ksp_max_it": 2000})
        uh = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="poisson_fallback_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
    uh.x.scatter_forward()
    grid_u = _sample_on_grid(uh, domain, nx_out, ny_out, bbox)
    verification = _manufactured_solution_check(comm)
    elapsed = time.perf_counter() - t0
    ksp = problem.solver
    solver_info = {"mesh_resolution": int(mesh_resolution), "element_degree": int(element_degree), "ksp_type": str(ksp.getType()), "pc_type": str(ksp.getPC().getType()), "rtol": float(rtol), "iterations": int(ksp.getIterationNumber()), "wall_time_sec_estimate": float(elapsed)}
    solver_info.update(verification)
    if comm.rank == 0:
        return {"u": np.asarray(grid_u, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}, "pde": {"time": None}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
