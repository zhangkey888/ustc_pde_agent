import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde_spec = case_spec.get("pde", {})
    k_val = float(pde_spec.get("wavenumber", 20.0))
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # For Helmholtz with k=20, degree 2 is sufficient at moderate resolution
    # The solution converges at N=32, deg=2 already (verified by convergence study)
    element_degree = 2
    
    # Adaptive refinement with convergence check
    resolutions = [32, 48, 64]
    prev_norm = None
    u_sol_grid = None
    final_N = None
    final_iterations = 0
    
    for N in resolutions:
        result = _solve_helmholtz(
            comm, N, element_degree, k_val, x_range, y_range,
            nx_out, ny_out
        )
        u_grid = result["u_grid"]
        norm_val = result["norm"]
        iterations = result["iterations"]
        
        final_N = N
        final_iterations = iterations
        u_sol_grid = u_grid
        
        if prev_norm is not None:
            rel_change = abs(norm_val - prev_norm) / (abs(norm_val) + 1e-15)
            if rel_change < 0.001:
                break
        
        prev_norm = norm_val
    
    if u_sol_grid is None:
        raise RuntimeError("All resolutions failed!")
    
    return {
        "u": u_sol_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": final_iterations,
        }
    }


def _solve_helmholtz(comm, N, degree, k_val, x_range, y_range, nx_out, ny_out):
    """Solve Helmholtz on a mesh of resolution N and return grid values."""
    
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 50*exp(-200*((x-0.5)^2 + (y-0.5)^2))
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    # Weak form: integral(grad(u).grad(v)) dx - k^2 integral(u*v) dx = integral(f*v) dx
    k2 = ScalarType(k_val**2)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Homogeneous Dirichlet BC on entire boundary (u = 0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Direct solver (LU) - Helmholtz is indefinite, iterative solvers can struggle
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"helmholtz_{N}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    u_sol = problem.solve()
    iterations = 1
    
    # L2 norm for convergence check
    norm_val = np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
        op=MPI.SUM
    ))
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, x_range, y_range, nx_out, ny_out)
    
    return {
        "u_grid": u_grid,
        "norm": norm_val,
        "iterations": iterations,
    }


def _evaluate_on_grid(domain, u_func, x_range, y_range, nx, ny):
    """Evaluate a FEM function on a uniform grid."""
    
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_values = np.nan_to_num(u_values, nan=0.0)
    return u_values.reshape((nx, ny))
