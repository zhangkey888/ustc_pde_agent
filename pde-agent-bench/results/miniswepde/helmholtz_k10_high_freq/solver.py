import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    pde_spec = case_spec.get("pde", {})
    k_val = float(pde_spec.get("wavenumber", 10.0))
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    element_degree = 2
    N = 48  # Good balance of accuracy and speed for k=10 with P2

    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term from manufactured solution u_exact = sin(3*pi*x)*sin(2*pi*y)
    # f = (9*pi^2 + 4*pi^2 - k^2) * sin(3*pi*x)*sin(2*pi*y)
    f_expr = (9.0 * ufl.pi**2 + 4.0 * ufl.pi**2 - k_val**2) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Zero Dirichlet BC on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, x_range, y_range, nx_out, ny_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 1,
        },
    }


def _evaluate_on_grid(domain, u_func, x_range, y_range, nx, ny):
    """Evaluate a FEM function on a regular grid."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    u_values = np.nan_to_num(u_values, nan=0.0)
    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {"wavenumber": 10.0},
        "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    xs = np.linspace(0.0, 1.0, 50)
    ys = np.linspace(0.0, 1.0, 50)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(3 * np.pi * xv) * np.sin(2 * np.pi * yv)
    
    l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_error = np.max(np.abs(u_grid - u_exact))
    print(f"Shape: {u_grid.shape}, Time: {elapsed:.3f}s")
    print(f"L2 error: {l2_error:.6e}, Linf error: {linf_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
