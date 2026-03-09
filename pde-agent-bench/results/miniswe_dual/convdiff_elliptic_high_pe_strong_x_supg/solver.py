import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [15.0, 0.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    N = 48
    degree = 2
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term from manufactured solution u = sin(pi*x)*sin(pi*y)
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian(u) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # f = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*sin(pi*x)*pi*cos(pi*y)
    f_expr = (2.0 * epsilon * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.sin(ufl.pi * x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1]))
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm + 1e-10)
    
    # SUPG test function modification
    v_stab = ufl.dot(beta, ufl.grad(v))
    
    # Bilinear form with SUPG
    a_form = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
              + tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), v_stab) * ufl.dx)
    
    # Linear form with SUPG
    L_form = (f_expr * v * ufl.dx
              + tau * f_expr * v_stab * ufl.dx)
    
    # Boundary conditions (u=0 on all boundaries for sin(pi*x)*sin(pi*y))
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
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


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.01,
                "beta": [15.0, 0.0],
            }
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    
    # Check against exact solution on grid
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact_grid = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    grid_error = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact_grid))
    print(f"Grid RMS error: {grid_error:.6e}")
    print(f"Grid max error: {max_error:.6e}")
    print(f"Iterations: {result['solver_info']['iterations']}")
