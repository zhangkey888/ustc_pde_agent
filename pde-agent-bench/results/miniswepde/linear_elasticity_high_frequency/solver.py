import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Parse parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    E_val = float(params.get("E", 1.0))
    nu_val = float(params.get("nu", 0.28))
    
    # Lame parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    
    # P4 elements with moderate mesh: excellent accuracy for high-frequency solution
    degree = 4
    N = 48
    ksp_used = "cg"
    pc_used = "hypre"
    rtol_used = 1e-12
    
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution as UFL expression
    u_exact_0 = ufl.sin(4*pi*x[0]) * ufl.sin(3*pi*x[1])
    u_exact_1 = ufl.cos(3*pi*x[0]) * ufl.sin(4*pi*x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu_val * epsilon(u) + lam_val * ufl.tr(epsilon(u)) * ufl.Identity(2)
    
    # Source term: -div(sigma(u_exact)) = f
    f = -ufl.div(sigma(u_exact))
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve with CG + AMG (hypre)
    iterations = 0
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_used,
                "pc_type": pc_used,
                "ksp_rtol": str(rtol_used),
                "ksp_max_it": "3000",
            },
            petsc_options_prefix="elast_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_used = "preonly"
        pc_used = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_used,
                "pc_type": pc_used,
            },
            petsc_options_prefix="elast_"
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    x_pts = np.linspace(x_range[0], x_range[1], nx_out)
    y_pts = np.linspace(y_range[0], y_range[1], ny_out)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = xx.ravel()
    points_3d[:, 1] = yy.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate displacement (vector valued, 2 components)
    u_values = np.full((points_3d.shape[0], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :2]
    
    # Compute displacement magnitude
    disp_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = disp_mag.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_used,
            "pc_type": pc_used,
            "rtol": rtol_used,
            "iterations": iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "linear_elasticity",
            "parameters": {"E": 1.0, "nu": 0.28},
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
    
    t_start = time.time()
    result = solve(case_spec)
    t_end = time.time()
    
    print(f"Wall time: {t_end - t_start:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {result['u'].min():.6e}, max: {result['u'].max():.6e}")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify against exact solution on the grid
    x_pts = np.linspace(0, 1, 50)
    y_pts = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_pts, y_pts, indexing='ij')
    u0_exact = np.sin(4*np.pi*xx) * np.sin(3*np.pi*yy)
    u1_exact = np.cos(3*np.pi*xx) * np.sin(4*np.pi*yy)
    mag_exact = np.sqrt(u0_exact**2 + u1_exact**2)
    
    error = np.max(np.abs(result['u'] - mag_exact))
    print(f"Max pointwise error in displacement_magnitude: {error:.6e}")
    
    l2_err = np.sqrt(np.mean((result['u'] - mag_exact)**2))
    print(f"RMS error: {l2_err:.6e}")
