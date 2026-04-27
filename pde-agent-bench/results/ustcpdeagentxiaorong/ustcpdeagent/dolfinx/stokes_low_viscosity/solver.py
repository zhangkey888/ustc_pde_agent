import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    
    # Output grid specification
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    N = 128  # mesh resolution - increased for better accuracy
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Create mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi_val = ufl.pi
    
    # Manufactured solution
    u_exact = ufl.as_vector([
        pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Source term from manufactured solution: f = -nu * laplacian(u) + grad(p)
    nu_const = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = -nu_const * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Symmetric Stokes weak form
    a = (nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity BC from manufactured solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    
    p0_func = fem.Function(Q)
    p0_func.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Solve with direct LU (MUMPS)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    
    # Sample velocity magnitude onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_3d = np.zeros((nx_out * ny_out, 3))
    pts_3d[:, 0] = XX.ravel()
    pts_3d[:, 1] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    
    # Find cells for each point
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.zeros(nx_out * ny_out)
    
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_eval, cells_eval)
        magnitude = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitude[idx]
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    iterations = 1
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.1},
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Compare with exact solution at grid points
    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    
    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    error = np.abs(result['u'] - mag_exact)
    print(f"Max pointwise error in velocity magnitude: {error.max():.6e}")
    print(f"Mean pointwise error: {error.mean():.6e}")
    
    l2_err = np.sqrt(np.sum((result['u'] - mag_exact)**2) / np.sum(mag_exact**2))
    print(f"Relative L2 error on grid: {l2_err:.6e}")
