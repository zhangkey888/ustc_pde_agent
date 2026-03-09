import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse case spec
    pde = case_spec.get("pde", {})
    nx_out = case_spec.get("output", {}).get("nx", 50)
    ny_out = case_spec.get("output", {}).get("ny", 50)
    
    kappa_val = 1.0
    coeffs = pde.get("coefficients", {})
    if "kappa" in coeffs:
        kappa_val = float(coeffs["kappa"])
    
    N = 100
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    # -div(kappa * grad(u)) = f => f = 2*pi^2*kappa*sin(pi*x)*sin(pi*y)
    f_expr = 2.0 * ufl.pi**2 * kappa_val * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary condition: u = 0 on boundary (exact solution vanishes on unit square boundary)
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
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.zeros(points_3d.shape[0])
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            val = u_sol.eval(points_3d[i:i+1], [links[0]])
            u_values[i] = val.flatten()[0]
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
        },
        "domain": {},
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"RMS error: {error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"Grid shape: {u_grid.shape}")
