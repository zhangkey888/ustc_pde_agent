import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid parameters
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Agent choices
    nx = 64
    ny = 64
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])], [nx, ny], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = 0.2
    
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Manufactured source term
    def eps(u_f):
        return ufl.sym(ufl.grad(u_f))
    def sigma(u_f, p_f):
        return 2.0 * nu * eps(u_f) - p_f * ufl.Identity(gdim)
        
    f = ufl.grad(u_ex) * u_ex - ufl.div(2.0 * nu * eps(u_ex)) + ufl.grad(p_ex)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    J = ufl.derivative(F, w)

    # BCs
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    bc_u = fem.dirichletbc(u_bc, boundary_dofs_u, W.sub(0))
    bcs = [bc_u]

    # Pressure pinning
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        # Evaluate exact pressure at (0,0)
        p0.x.array[:] = 0.0 # sin(0)*cos(0) = 0
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    w.x.array[:] = 0.0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    
    # Try to solve directly with Newton
    try:
        n_iters, converged = problem.solve()
    except Exception as e:
        print(f"Newton solver failed: {e}")
        n_iters = 0

    w.x.scatter_forward()
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()

    # Interpolate output
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)

    u_vals_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        mags = np.linalg.norm(u_vals, axis=1)
        u_vals_mag[eval_map] = mags

    u_vals_mag = u_vals_mag.reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree_u,
        "ksp_type": petsc_options["ksp_type"],
        "pc_type": petsc_options["pc_type"],
        "rtol": petsc_options["snes_rtol"],
        "iterations": n_iters,
        "nonlinear_iterations": [n_iters],
    }

    return {"u": u_vals_mag, "solver_info": solver_info}
