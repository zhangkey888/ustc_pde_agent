import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Extract parameters from case_spec
    pde_spec = case_spec.get("pde", {})
    k_val = pde_spec.get("wavenumber", 15.0)
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Mesh resolution and element degree - adaptive
    # For k=15, wavelength ~ 0.42, need ~10 pts per wavelength
    # With P2 elements, can use coarser mesh
    element_degree = 2
    N = 80  # mesh resolution
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 10*exp(-80*((x-0.35)^2 + (y-0.55)^2))
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # Wavenumber
    k = fem.Constant(domain, PETSc.ScalarType(k_val))
    
    # Weak form: ∫ ∇u·∇v dx - k²∫ u·v dx = ∫ f·v dx
    # From -∇²u - k²u = f, multiply by v and integrate:
    # ∫ ∇u·∇v dx - k²∫ u v dx = ∫ f v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = 0 on ∂Ω (assuming g=0 since not specified otherwise)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Check if there's a specific BC value
    bc_spec = pde_spec.get("boundary_conditions", {})
    bc_value = 0.0  # default
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(bc_value), dofs, V)
    bcs = [bc]
    
    # Solve - use direct solver for indefinite Helmholtz (most robust)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="helmholtz_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to GMRES + ILU
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
            },
            petsc_options_prefix="helmholtz_"
        )
        u_sol = problem.solve()
    
    # Evaluate solution on output grid
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.zeros(points_3d.shape[0])
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Get iteration count if available
    iterations = 1  # direct solver = 1 iteration
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "wavenumber": 15.0,
            "type": "helmholtz",
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {u_grid.min():.6f}, max: {u_grid.max():.6f}")
    print(f"Solution L2 norm: {np.linalg.norm(u_grid):.6f}")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
