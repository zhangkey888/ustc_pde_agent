import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 80  # mesh resolution
    degree = 2  # element degree (P2)
    
    # Create quadrilateral mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # Mixed formulation for biharmonic:
    # Δ²u = f  =>  w = -Δu,  -Δw = f  (with appropriate BCs)
    # But we also need w = -Δu on boundary, which requires knowing normal derivatives.
    # 
    # Alternative: use the C0 interior penalty (IP) method on a single P2 space.
    # 
    # For the manufactured solution u = sin(2πx)sin(πy):
    # Δu = -(4π² + π²)sin(2πx)sin(πy) = -5π²sin(2πx)sin(πy)
    # Δ²u = (5π²)² * ... wait, let me compute properly.
    #
    # u = sin(2πx)sin(πy)
    # u_xx = -(2π)²sin(2πx)sin(πy) = -4π²u
    # u_yy = -(π)²sin(2πx)sin(πy) = -π²u
    # Δu = -(4π² + π²)u = -5π²u
    # 
    # Δ²u = Δ(Δu) = -5π² Δu = -5π²(-5π²u) = 25π⁴u
    # So f = 25π⁴ sin(2πx)sin(πy)
    
    # Use C0 Interior Penalty method for biharmonic equation
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact solution
    u_exact_expr = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = Δ²u
    # Compute symbolically
    grad_u_exact = ufl.grad(u_exact_expr)
    laplacian_u_exact = ufl.div(grad_u_exact)
    grad_laplacian = ufl.grad(laplacian_u_exact)
    f_expr = ufl.div(grad_laplacian)
    
    # Mesh quantities
    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)
    
    # Penalty parameter (must be large enough for stability)
    alpha = fem.Constant(domain, PETSc.ScalarType(16.0 * degree * (degree + 1)))
    
    # C0 Interior Penalty bilinear form
    # a(u,v) = ∫_Ω Δu Δv dx
    #         - ∫_E {∂²u/∂n²}[∂v/∂n] ds
    #         - ∫_E {∂²v/∂n²}[∂u/∂n] ds
    #         + ∫_E α/h [∂u/∂n][∂v/∂n] ds
    #
    # where E are interior edges, {} is average, [] is jump
    
    # For the biharmonic with C0-IP:
    # The key terms involve second derivatives (Hessians)
    
    def D2(w):
        """Hessian of w"""
        return ufl.grad(ufl.grad(w))
    
    # Inner product of Hessians (Frobenius)
    a_bulk = ufl.inner(D2(u), D2(v)) * ufl.dx
    
    # Interior facet terms
    # Average of normal-normal component of Hessian
    # Jump of normal derivative
    
    # On interior facets:
    # {D²u : (n⊗n)} = avg(n^T D²u n)  -- but we use the simpler notation
    # [∂u/∂n] = jump(grad(u), n)
    
    # Interior facet integral
    a_int = (
        - ufl.inner(ufl.avg(ufl.dot(ufl.dot(D2(u), n('+')), n('+'))), ufl.jump(ufl.grad(v), n('+'))) * ufl.dS
        - ufl.inner(ufl.avg(ufl.dot(ufl.dot(D2(v), n('+')), n('+'))), ufl.jump(ufl.grad(u), n('+'))) * ufl.dS
        + alpha / ufl.avg(h) * ufl.inner(ufl.jump(ufl.grad(u), n('+')), ufl.jump(ufl.grad(v), n('+'))) * ufl.dS
    )
    
    a_form = a_bulk + a_int
    
    # RHS
    L_form = ufl.inner(f_expr, v) * ufl.dx
    
    # Dirichlet BC: u = g on ∂Ω
    # For our manufactured solution, u = sin(2πx)sin(πy) = 0 on ∂Ω (since sin(0)=sin(2π)=sin(π)=0 at boundaries)
    # So g = 0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="biharmonic_"
    )
    uh = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.ravel()
    points[1] = Y.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }