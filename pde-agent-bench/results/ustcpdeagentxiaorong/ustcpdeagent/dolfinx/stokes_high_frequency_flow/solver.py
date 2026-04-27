import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    
    nu_val = case_spec["pde"]["viscosity"]
    
    # Choose mesh resolution - high frequency solution needs good resolution
    # The manufactured solution has 2*pi frequency, so we need decent resolution
    N = 80  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W_map = W.sub(0).collapse()
    Q, Q_to_W_map = W.sub(1).collapse()
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(msh)
    
    # Manufactured solution
    # u_exact = [2*pi*cos(2*pi*y)*sin(2*pi*x), -2*pi*cos(2*pi*x)*sin(2*pi*y)]
    # p_exact = sin(2*pi*x)*cos(2*pi*y)
    pi = ufl.pi
    
    u_exact = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
        -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_exact = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])
    
    # Compute source term: f = -nu * laplacian(u) + grad(p)
    # For the manufactured solution:
    # u1 = 2*pi*cos(2*pi*y)*sin(2*pi*x)
    # u1_xx = 2*pi*cos(2*pi*y)*(-4*pi^2)*sin(2*pi*x) = -8*pi^3*cos(2*pi*y)*sin(2*pi*x)
    # u1_yy = 2*pi*(-4*pi^2)*cos(2*pi*y)*sin(2*pi*x) = -8*pi^3*cos(2*pi*y)*sin(2*pi*x) [wait, let me recalculate]
    # Actually let me just use UFL to compute it symbolically
    
    # -nu * div(grad(u_exact)) + grad(p_exact) = f
    # We need the strong form source
    # Let's compute using UFL
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Bilinear form: Stokes equations
    # -nu * laplacian(u) + grad(p) = f  =>  nu * (grad(u), grad(v)) - (p, div(v)) = (f, v)
    # div(u) = 0                        =>  (div(u), q) = 0
    nu_const = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    a = (nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    
    # All boundaries
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity BC from exact solution
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin at (0,0) since all Dirichlet velocity BCs
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    
    # p_exact at (0,0) = sin(0)*cos(0) = 0
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Solve
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    
    # Extract velocity and pressure
    u_h = w_h.sub(0).collapse()
    p_h = w_h.sub(1).collapse()
    
    # Compute error for verification
    error_u = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_L2_u = np.sqrt(comm.allreduce(fem.assemble_scalar(error_u), op=MPI.SUM))
    print(f"Velocity L2 error: {error_L2_u:.6e}")
    
    error_p = fem.form(ufl.inner(p_h - p_exact, p_h - p_exact) * ufl.dx)
    error_L2_p = np.sqrt(comm.allreduce(fem.assemble_scalar(error_p), op=MPI.SUM))
    print(f"Pressure L2 error: {error_L2_p:.6e}")
    
    # Sample velocity magnitude onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    # Build bounding box tree for point evaluation
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full((ny_out * nx_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_arr, cells_arr)  # shape (N, gdim)
        magnitude = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitude[idx]
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Handle any NaN values at boundary points by using nearest neighbor
    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.ravel())
        if np.any(valid):
            coords_valid = np.column_stack([XX.ravel()[valid], YY.ravel()[valid]])
            interp = NearestNDInterpolator(coords_valid, u_grid.ravel()[valid])
            nan_mask = np.isnan(u_grid.ravel())
            coords_nan = np.column_stack([XX.ravel()[nan_mask], YY.ravel()[nan_mask]])
            u_grid.ravel()[nan_mask] = interp(coords_nan)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,  # Not easily accessible from LinearProblem
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    # Test with a mock case_spec
    case_spec = {
        "pde": {
            "viscosity": 1.0,
        },
        "output": {
            "field": "velocity_magnitude",
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
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Min magnitude: {np.nanmin(result['u']):.6f}")
    print(f"Max magnitude: {np.nanmax(result['u']):.6f}")
