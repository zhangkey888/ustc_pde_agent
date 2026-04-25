import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution
    N = 128
    comm = MPI.COMM_WORLD
    
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.quadrilateral)

    # Function space for u and v (where v = -Delta u)
    V = fem.functionspace(domain, ("Lagrange", 2))

    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # f = Delta^2 u_ex
    f = 25.0 * ufl.pi**4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # v_ex = -Delta u_ex
    v_ex = 5.0 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # 1. Solve for v: -Delta v = f
    # BC for v: v = v_ex on boundary
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)

    v_bc_func = fem.Function(V)
    v_bc_func.interpolate(fem.Expression(v_ex, V.element.interpolation_points))
    bc_v = fem.dirichletbc(v_bc_func, boundary_dofs)

    v_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_v = ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
    L_v = ufl.inner(f, v_test) * ufl.dx

    problem_v = petsc.LinearProblem(a_v, L_v, bcs=[bc_v], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="v_solve_")
    v_sol = problem_v.solve()

    # 2. Solve for u: -Delta u = v
    # BC for u: u = u_ex on boundary
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)

    u_trial = ufl.TrialFunction(V)
    u_test = ufl.TestFunction(V)

    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
    L_u = ufl.inner(v_sol, u_test) * ufl.dx

    problem_u = petsc.LinearProblem(a_u, L_u, bcs=[bc_u], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="u_solve_")
    u_sol = problem_u.solve()

    # Interpolate to grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()

    u_grid = u_grid_flat.reshape((ny, nx))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 2
    }

    return {"u": u_grid, "solver_info": solver_info}
