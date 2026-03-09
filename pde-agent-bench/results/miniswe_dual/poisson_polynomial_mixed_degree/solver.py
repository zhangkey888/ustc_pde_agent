import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    resolutions = [32, 64, 128]
    element_degree = 2
    prev_norm = None
    final_result = None
    for N in resolutions:
        result = _solve_at_resolution(N, element_degree, case_spec)
        curr_norm = result["norm"]
        if prev_norm is not None:
            rel_change = abs(curr_norm - prev_norm) / (abs(curr_norm) + 1e-15)
            if rel_change < 0.01:
                return result["output"]
        prev_norm = curr_norm
        final_result = result["output"]
    return final_result


def _solve_at_resolution(N, degree, case_spec):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = x[0]*(1-x[0])*x[1]*(1-x[1])*(1 + 0.5*x[0]*x[1])
    kappa = fem.Constant(domain, ScalarType(1.0))
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
            petsc_options_prefix="poisson_")
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(a, L, bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="poisson_")
        u_sol = problem.solve()
    l2_norm = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)), op=MPI.SUM))
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    u_grid = _evaluate_function(u_sol, domain, points_3d)
    u_grid = u_grid.reshape((nx_out, ny_out))
    output = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type": ksp_type, "pc_type": pc_type,
            "rtol": rtol, "iterations": 0,
        }
    }
    return {"norm": l2_norm, "output": output}


def _evaluate_function(u_func, domain, points_3d):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    n_points = points_3d.shape[0]
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    return u_values
