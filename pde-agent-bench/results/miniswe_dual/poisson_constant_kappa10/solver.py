import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 10.0))
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Manufactured solution: u = sin(pi*x)*sin(2*pi*y)
    # f = -div(kappa * grad(u)) = kappa * (pi^2 + 4*pi^2) * sin(pi*x)*sin(2*pi*y)
    #   = kappa * 5*pi^2 * sin(pi*x)*sin(2*pi*y)
    
    # Adaptive mesh refinement
    comm = MPI.COMM_WORLD
    element_degree = 2
    N = 64  # Start with moderate resolution for degree 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    # Source term from manufactured solution
    f_expr = kappa_val * 5.0 * (np.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = sin(pi*x)*sin(2*pi*y) = 0 on boundary of [0,1]^2
    # Since sin(pi*0)=0, sin(pi*1)=0, sin(2*pi*0)=0, sin(2*pi*1)=0
    # So g = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,  # Not easily accessible from LinearProblem
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 10.0},
        },
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    
    # Compare with exact solution
    nx_out, ny_out = 50, 50
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.4f}s")
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
