import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    # Extract parameters
    pde_spec = case_spec.get("pde", {})
    k_val = float(pde_spec.get("wavenumber", 12.0))
    k2 = k_val * k_val
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    
    # For k=12 with P2 elements, N=32 gives good accuracy
    N = 32
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(x)*cos(2*pi*y)
    # Laplacian: nabla^2 u = exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    # f = -nabla^2 u - k^2 u = exp(x)*cos(2*pi*y)*(4*pi^2 - 1 - k^2)
    f_expr = ufl.exp(x[0]) * ufl.cos(2.0 * ufl.pi * x[1]) * (4.0 * ufl.pi**2 - 1.0 - k2)
    
    # Bilinear form: a(u,v) = integral(grad(u).grad(v) - k^2*u*v) dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = exp(x)*cos(2*pi*y) on boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.exp(xx[0]) * np.cos(2.0 * np.pi * xx[1]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve with direct solver (robust for indefinite Helmholtz)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        },
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {"type": "helmholtz", "wavenumber": 12.0},
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.exp(X) * np.cos(2.0 * np.pi * Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"PASS accuracy: {error < 3.49e-3}")
    print(f"PASS time: {elapsed < 10.27}")
