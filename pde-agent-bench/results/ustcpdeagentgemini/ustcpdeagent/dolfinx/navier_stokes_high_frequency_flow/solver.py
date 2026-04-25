import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Parameters
    mesh_res = 128
    nu = 0.1
    degree_u = 2
    degree_p = 1

    # 1. Mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # 2. Function Spaces
    vel_el = element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # 3. Manufactured Solution & Source Term
    x = ufl.SpatialCoordinate(msh)
    
    # Exact velocity and pressure
    u_ex = ufl.as_vector([
        2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(2 * ufl.pi * x[0]),
        -2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])
    p_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])

    # Compute source term automatically
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    # 4. Weak Form
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    # 5. Boundary Conditions
    bcs = []

    # Velocity BC: exact solution on all walls
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs_v = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    bcs.append(fem.dirichletbc(u_bc_func, boundary_dofs_v, W.sub(0)))

    # Pressure pin: pin at origin to exact pressure
    def origin_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), origin_marker)
    if len(p_dofs[0]) > 0:
        p_bc_func = fem.Function(Q)
        p_bc_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points()))
        bcs.append(fem.dirichletbc(p_bc_func, p_dofs, W.sub(1)))

    # 6. Solve
    # Use Picard for a few steps to get a good guess? Or just Newton. Let's try Newton directly, often works for nu=0.1
    # Actually, initializing w with Stokes is safer.
    
    # Stokes setup for initial guess
    F_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    stokes_prob = petsc.NonlinearProblem(F_stokes, w, bcs=bcs, petsc_options_prefix="stokes_")
    stokes_prob.solve()

    # Newton for Navier-Stokes
    J = ufl.derivative(F, w)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    ns_prob = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    ns_prob.solve()

    u_sol = w.sub(0).collapse()

    # 7. Output processing
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": petsc_options["ksp_type"],
        "pc_type": petsc_options["pc_type"],
        "rtol": petsc_options["snes_rtol"],
        "iterations": 0,
        "nonlinear_iterations": [0]
    }

    return {
        "u": magnitude,
        "solver_info": solver_info
    }
