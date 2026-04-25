import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # PDE parameters
    eps = 0.02
    beta = np.array([6.0, 2.0])
    beta_norm = np.linalg.norm(beta)
    
    # Output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Solver parameters
    mesh_res = 256
    element_degree = 2
    rtol = 1e-10
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate for UFL expressions
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(u_exact) = -8*pi^2*sin(2*pi*x)*sin(2*pi*y)
    # beta . grad(u_exact) = 12*pi*cos(2*pi*x)*sin(2*pi*y) + 4*pi*sin(2*pi*x)*cos(2*pi*y)
    f_val = (eps * 8.0 * ufl.pi**2 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
             + 12.0 * ufl.pi * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
             + 4.0 * ufl.pi * ufl.sin(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1]))
    
    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    beta_ufl = ufl.as_vector(beta)
    h = ufl.CellDiameter(domain)
    
    # SUPG stabilization parameter
    # For high Pe: tau ~ h/(2|beta|), for low Pe: tau ~ h^2/(4*eps)
    # Smooth blend: tau = h / (2*|beta| + 4*eps/h)
    # In UFL: use min_value approach
    Pe_h = beta_norm * h / (2.0 * eps)
    tau = h / (2.0 * beta_norm) * ufl.min_value(Pe_h, 1.0)
    # For Pe_h >> 1: tau = h/(2|beta|) (convection-dominated)
    # For Pe_h << 1: tau = h^2/(4*eps) (diffusion-dominated)
    
    # Bilinear form: Galerkin + SUPG
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
         + tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), ufl.dot(beta_ufl, ufl.grad(v))) * ufl.dx)
    
    # Linear form: Galerkin + SUPG
    L = (f_val * v * ufl.dx
         + tau * f_val * ufl.dot(beta_ufl, ufl.grad(v)) * ufl.dx)
    
    # Solve with direct LU
    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                   petsc_options_prefix="convdiff_")
    u_sol = problem.solve()
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    error_diff = u_sol.x.petsc_vec - u_exact_func.x.petsc_vec
    l2_error_sq = fem.assemble_scalar(fem.form((u_sol - u_exact_ufl)**2 * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_error_sq, op=MPI.SUM))
    
    h1_error_sq = fem.assemble_scalar(fem.form(
        ufl.inner(ufl.grad(u_sol - u_exact_ufl), ufl.grad(u_sol - u_exact_ufl)) * ufl.dx
        + (u_sol - u_exact_ufl)**2 * ufl.dx))
    h1_error = np.sqrt(comm.allreduce(h1_error_sq, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}, H1 error: {h1_error:.6e}")
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather results in parallel
    if comm.size > 1:
        from mpi4py import MPI as MPI2
        recv_buf = np.zeros_like(u_values)
        comm.Allreduce(u_values, recv_buf, op=MPI2.MAX)
        # Replace NaN with -inf so MAX works
        u_values_send = np.where(np.isnan(u_values), -np.inf, u_values)
        recv_buf = np.zeros_like(u_values_send)
        comm.Allreduce(u_values_send, recv_buf, op=MPI2.MAX)
        u_values = recv_buf
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": 1,
        }
    }
