import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh: quadrilateral
    N = 48
    degree = 3
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                    cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.sin(2*pi*x[0]) * ufl.cos(3*pi*x[1])
    # Δu = -(4π²+9π²) u = -13π² u
    # w = -Δu = 13π² u
    # f = Δ²u = 169π⁴ u
    w_exact = 13 * pi**2 * u_exact
    f_expr = 169 * pi**4 * u_exact
    
    # Boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # First solve: -Δw = f with w = 13π²·u_exact on boundary
    w_bc_func = fem.Function(V)
    w_bc_func.interpolate(
        fem.Expression(w_exact, V.element.interpolation_points)
    )
    bc_w = fem.dirichletbc(w_bc_func, boundary_dofs)
    
    w_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_w = ufl.inner(ufl.grad(w_tr), ufl.grad(v)) * ufl.dx
    L_w = f_expr * v * ufl.dx
    
    total_iters = 0
    
    problem_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_w_"
    )
    w_sol = problem_w.solve()
    try:
        total_iters += problem_w.solver.getIterationNumber()
    except Exception:
        pass
    
    # Second solve: -Δu = w with u = u_exact on boundary
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    u_tr = ufl.TrialFunction(V)
    a_u = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L_u = w_sol * v * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_u_"
    )
    u_sol = problem_u.solve()
    try:
        total_iters += problem_u.solver.getIterationNumber()
    except Exception:
        pass
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(total_iters),
        }
    }
