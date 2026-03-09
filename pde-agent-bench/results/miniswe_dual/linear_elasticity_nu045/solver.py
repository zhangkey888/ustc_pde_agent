import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    E_val = float(params.get("E", 1.0))
    nu_val = float(params.get("nu", 0.45))
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Lame parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    # Manufactured solution
    # u_exact = [sin(pi*x)*sin(pi*y), cos(pi*x)*sin(pi*y)]
    
    # Adaptive mesh refinement
    element_degree = 2
    N = 112  # Start with a reasonable mesh for degree 2
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    # Vector function space
    V = fem.functionspace(domain, ("Lagrange", element_degree, (domain.geometry.dim,)))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Material parameters as UFL constants
    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lam = fem.Constant(domain, PETSc.ScalarType(lam_val))
    
    # Exact solution (UFL)
    u_exact = ufl.as_vector([
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(domain.geometry.dim)
    
    # Compute source term from manufactured solution
    f = -ufl.div(sigma(u_exact))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions - apply exact solution on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "2000",
            },
            petsc_options_prefix="elasticity_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="elasticity_"
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full((points_3d.shape[0], 2), np.nan)
    
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
        for idx, i in enumerate(eval_map):
            u_values[i, :] = vals[idx, :]
    
    # Compute displacement magnitude
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,  # placeholder
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "linear_elasticity",
            "parameters": {"E": 1.0, "nu": 0.45},
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
        "output": {
            "nx": 50,
            "ny": 50,
            "field": "displacement_magnitude",
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Min: {np.nanmin(u_grid):.6e}, Max: {np.nanmax(u_grid):.6e}")
    print(f"Time: {elapsed:.3f}s")
    
    # Compute exact displacement magnitude on the same grid
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    ux_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    uy_exact = np.cos(np.pi * X) * np.sin(np.pi * Y)
    u_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    error = np.max(np.abs(u_grid - u_mag_exact))
    l2_error = np.sqrt(np.mean((u_grid - u_mag_exact)**2))
    print(f"Max error: {error:.6e}")
    print(f"L2 error: {l2_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
