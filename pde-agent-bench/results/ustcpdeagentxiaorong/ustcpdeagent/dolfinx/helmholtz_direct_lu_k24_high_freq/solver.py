import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Extract parameters
    k_val = case_spec["pde"]["params"]["k"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Use P2 elements with moderate mesh for speed
    # k=24, need good resolution but P2 should be sufficient
    degree = 2
    mesh_res = 64
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(5 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term: -∇²u - k²u = f
    # ∇²u = -(25π² + 16π²) * sin(5πx)*sin(4πy) = -41π² * u_exact
    # f = -∇²u - k²u = (41π² - k²) * u_exact
    k2 = fem.Constant(domain, ScalarType(k_val**2))
    
    f = (41.0 * ufl.pi**2 - k2) * u_exact
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with direct LU (robust for indefinite Helmholtz)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0, :] = XX.ravel()
    pts[1, :] = YY.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
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
    import time
    
    case_spec = {
        "pde": {
            "params": {"k": 24.0}
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
    print(f"u_grid shape: {result['u'].shape}")
    
    # Compute point-wise error against exact
    nx, ny = 100, 100
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.sin(5*np.pi*XX) * np.sin(4*np.pi*YY)
    
    err = np.abs(result['u'] - u_exact_grid)
    l2_grid_err = np.sqrt(np.nanmean(err**2))
    linf_err = np.nanmax(err)
    print(f"Grid L2 error: {l2_grid_err:.6e}")
    print(f"Grid Linf error: {linf_err:.6e}")
