import os; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Mesh
    nx_mesh = 64
    ny_mesh = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    # 2. Function Spaces (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # 3. Exact Solution & Source Term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    ])
    p_exact = ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    nu = 1.0
    # Auto-derive forcing
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # 4. Weak Form
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # 5. Boundary Conditions
    fdim = domain.topology.dim - 1
    # Full boundary
    wall_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_c: np.ones(x_c.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    
    u_bc = fem.Function(V)
    u_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_expr)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin (optional but good practice for full Dirichlet velocity)
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x_c: np.isclose(x_c[0], 0.0) & np.isclose(x_c[1], 0.0))
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        # Interpolate exact pressure at the pinned point
        p_expr = fem.Expression(p_exact, Q.element.interpolation_points)
        p0.interpolate(p_expr)
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # 6. Solve
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_sol = problem.solve()
    u_sol, p_sol = w_sol.sub(0).collapse(), w_sol.sub(1).collapse()

    # 7. Output interpolation
    out_grid = case_spec["output"]["grid"]
    nx, ny = out_grid["nx"], out_grid["ny"]
    bbox = out_grid["bbox"]
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
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_eval = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_eval[eval_map] = vals
    
    # Calculate magnitude
    u_mag = np.linalg.norm(u_eval, axis=1).reshape(ny, nx)

    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
