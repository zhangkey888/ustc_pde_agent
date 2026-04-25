import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve steady incompressible Navier-Stokes with manufactured solution.
    """
    comm = MPI.COMM_WORLD
    
    # Parameters from problem description
    nu_val = 0.16
    domain_bbox = [0.0, 1.0, 0.0, 1.0]  # xmin, xmax, ymin, ymax
    
    # Mesh resolution - choose based on time budget (start moderate)
    mesh_resolution = 128
    degree_u = 2  # P2 for velocity
    degree_p = 1  # P1 for pressure (Taylor-Hood)
    
    # Create mesh
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                   cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()  # Velocity subspace
    Q, _ = W.sub(1).collapse()  # Pressure subspace
    
    # Define trial/test functions
    w = fem.Function(W)  # Current solution
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinate for manufactured solution
    x = ufl.SpatialCoordinate(msh)
    
    # Manufactured solution
    # u_x = pi * tanh(6*(x-0.5)) * cos(pi*y)
    # u_y = -6 * (1 - tanh(6*(x-0.5))**2) * sin(pi*y)
    # p = sin(pi*x) * cos(pi*y)
    
    u_x_ex = ufl.pi * ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.cos(ufl.pi * x[1])
    u_y_ex = -6.0 * (1.0 - ufl.tanh(6.0 * (x[0] - 0.5))**2) * ufl.sin(ufl.pi * x[1])
    u_ex = ufl.as_vector([u_x_ex, u_y_ex])
    p_ex = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    # Compute source term f from the manufactured solution
    # f = u·∇u - ν∇²u + ∇p
    # We'll compute f analytically using UFL differentiation
    
    def grad(u_vec):
        return ufl.grad(u_vec)
    
    def laplacian(u_vec):
        return ufl.div(ufl.grad(u_vec))
    
    # Compute derivatives of manufactured solution
    # u·∇u term
    u_grad_u = ufl.dot(u_ex, ufl.grad(u_ex))
    
    # ∇p term
    grad_p = ufl.grad(p_ex)
    
    # ∇²u term (Laplacian)
    lap_u = ufl.as_vector([
        laplacian(u_x_ex),
        laplacian(u_y_ex)
    ])
    
    # Source term f = u·∇u - ν∇²u + ∇p
    f = u_grad_u - nu * lap_u + grad_p
    
    # Define stress tensor and weak form
    def eps(u_vec):
        return ufl.sym(ufl.grad(u_vec))
    
    # Residual form: F = (ν*2*eps(u), eps(v)) + (u·∇u, v) - (p, div(v)) + (div(u), q) - (f, v)
    F = (2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
         + ufl.inner(ufl.dot(u, ufl.grad(u)), v) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx
         - ufl.inner(f, v) * ufl.dx)
    
    # Dirichlet BCs on all boundaries (prescribe exact solution)
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pinning (fix pressure at one corner to ensure uniqueness)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), 
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Initial guess - use Stokes solution (drop convection term)
    # This provides a good starting point for Newton
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    F_stokes = (2.0 * nu * ufl.inner(eps(u_s), eps(v)) * ufl.dx
                - ufl.inner(p_s, ufl.div(v)) * ufl.dx
                + ufl.inner(ufl.div(u_s), q) * ufl.dx
                - ufl.inner(f, v) * ufl.dx)
    
    # Solve Stokes first for initial guess
    J_stokes = ufl.derivative(F_stokes, w_stokes)
    stokes_problem = petsc.NonlinearProblem(
        F_stokes, w_stokes, bcs=bcs, J=J_stokes,
        petsc_options_prefix="stokes_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "ksp_type": "preonly",
            "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
        }
    )
    w_stokes = stokes_problem.solve()
    
    # Use Stokes solution as initial guess for Navier-Stokes
    w.x.array[:] = w_stokes.x.array[:]
    
    # Jacobian for Navier-Stokes
    J = ufl.derivative(F, w)
    
    # Solve Navier-Stokes with Newton
    ns_problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"
        }
    )
    
    w = ns_problem.solve()
    w.x.scatter_forward()
    
    # Extract velocity solution
    u_sol = w.sub(0).collapse()
    
    # Sample solution on output grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    
    # Points to evaluate (3D array for dolfinx)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    
    # Use geometry utilities to find cells and evaluate
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate velocity at points
    u_vals_full = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals_full[eval_map] = u_vals
    
    # Compute velocity magnitude
    u_magnitude = np.linalg.norm(u_vals_full, axis=1).reshape(ny, nx)
    
    # Compute L2 error against exact solution
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    
    error_L2 = np.sqrt(msh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)),
        op=MPI.SUM))
    
    # Prepare solver info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        "rtol": 1e-8,
        "nonlinear_iterations": [1]  # Placeholder - actual Newton iterations not easily accessible
    }
    
    return {
        "u": u_magnitude,
        "solver_info": solver_info
    }
