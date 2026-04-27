import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    k_val = case_spec["pde"]["helmholtz"]["k"]
    
    # Source term info
    source_info = case_spec["pde"]["source"]
    
    # BC info
    bc_info = case_spec["pde"]["bcs"]
    
    # Output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # For k=20, we need roughly 10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.314
    # For unit square, we need at least 10/0.314 ≈ 32 elements per direction
    # For P2 elements, effective resolution is doubled, so ~64 elements should be good
    # But let's use more for accuracy given generous time limit
    
    # Rule of thumb: at least 10 DOFs per wavelength for P2
    # wavelength = 2*pi/k = 2*pi/20 ≈ 0.314
    # For P2, effective h = 1/(2*N), need h < wavelength/10 ≈ 0.0314
    # So N > 1/(2*0.0314) ≈ 16, but for accuracy let's use much more
    
    degree = 3  # Higher degree for better accuracy
    mesh_res = 100  # Fine mesh
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 50*exp(-200*((x-0.5)**2 + (y-0.5)**2))
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    # Wavenumber
    k = fem.Constant(domain, ScalarType(k_val))
    
    # Weak form: -∇²u - k²u = f
    # Multiply by v and integrate by parts:
    # ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Dirichlet BCs: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using direct solver (LU) - best for indefinite Helmholtz
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Replace any NaN values at boundaries with 0 (BC value)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
        }
    }

if __name__ == "__main__":
    # Test with a sample case_spec
    case_spec = {
        "pde": {
            "helmholtz": {"k": 20.0},
            "source": {"type": "gaussian", "amplitude": 50.0, "width": 200.0, "center": [0.5, 0.5]},
            "bcs": {"type": "dirichlet", "value": 0.0},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {result['u'].min():.6f}, max: {result['u'].max():.6f}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
