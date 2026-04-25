import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Problem parameters
    kappa = 1.0
    
    # Mesh resolution - need fine mesh for boundary layer (exp(5*x) grows fast near x=1)
    mesh_res = 80
    elem_degree = 3
    
    # Create quadrilateral mesh (as suggested by case name)
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                    cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(5*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = -div(kappa * grad(u)) = -kappa * laplacian(u)
    # laplacian(u) = 25*exp(5*x)*sin(pi*y) - pi^2*exp(5*x)*sin(pi*y)
    # f = kappa * (pi^2 - 25) * exp(5*x)*sin(pi*y)
    f_ufl = kappa * (ufl.pi**2 - 25.0) * ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Variational form: kappa * inner(grad(u), grad(v)) * dx = f * v * dx
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    # Boundary conditions: u = exp(5*x)*sin(pi*y) on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with CG + AMG (good for Poisson)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])  # shape (3, N)
    
    # Probe solution at grid points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), 
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Gather across processes if parallel
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        # Replace remaining NaN with 0 (points not on any proc)
        u_values_global = np.nan_to_num(u_values_global, nan=0.0)
        u_values = u_values_global
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    L2_error = fem.assemble_scalar(
        fem.form((u_sol - u_exact_ufl)**2 * ufl.dx))
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    # Test with default case spec
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
    print(f"u min/max: {result['u'].min():.6e} / {result['u'].max():.6e}")
    # Check against exact solution
    nx_out, ny_out = 64, 64
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(5*XX) * np.sin(np.pi * YY)
    error = np.abs(result['u'] - u_exact).max()
    print(f"Max pointwise error: {error:.6e}")
