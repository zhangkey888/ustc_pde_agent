import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    k_val = case_spec["pde"]["params"]["k"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Choose mesh resolution and element degree for accuracy
    # k=10 => need sufficient resolution per wavelength
    # Rule of thumb: at least 10 points per wavelength for P1, fewer for higher order
    # Wavelength ~ 2*pi/k ~ 0.628
    # For P3 elements, we can use fewer cells
    mesh_resolution = 80  # cells per direction
    element_degree = 3
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Manufactured solution: u = sin(3*pi*x)*sin(2*pi*y)
    # -∇²u - k²u = f
    # ∇²u = -(9π² + 4π²) sin(3πx)sin(2πy) = -13π² sin(3πx)sin(2πy)
    # -∇²u = 13π² sin(3πx)sin(2πy)
    # -k²u = -k² sin(3πx)sin(2πy)
    # f = (13π² - k²) sin(3πx)sin(2πy)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_expr = ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f_expr = (13.0 * pi**2 - k_val**2) * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Dirichlet BC: u = g on ∂Ω
    # For this manufactured solution, g = sin(3πx)sin(2πy) on boundary
    # On boundary of [0,1]², sin(3πx)sin(2πy) = 0 since either x=0,1 or y=0,1
    # sin(3π*0)=0, sin(3π*1)=0, sin(2π*0)=0, sin(2π*1)=0
    # So g = 0 everywhere on boundary
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using direct solver (LU) since Helmholtz is indefinite
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Compute L2 error for verification
    error_form = fem.form(ufl.inner(u_sol - u_exact_expr, u_sol - u_exact_expr) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    
    # Create a test case_spec
    case_spec = {
        "pde": {
            "params": {"k": 10.0}
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    # Verify against exact solution on the grid
    xs = np.linspace(0.0, 1.0, 100)
    ys = np.linspace(0.0, 1.0, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    
    mask = ~np.isnan(result['u'])
    l2_grid_error = np.sqrt(np.mean((result['u'][mask] - u_exact[mask])**2))
    linf_grid_error = np.max(np.abs(result['u'][mask] - u_exact[mask]))
    print(f"Grid L2 error: {l2_grid_error:.6e}")
    print(f"Grid Linf error: {linf_grid_error:.6e}")
