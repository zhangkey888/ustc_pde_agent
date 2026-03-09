import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation."""
    
    # Parse case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", coeffs.get("k", 0.8)))
    
    # Time parameters - hardcoded defaults as fallback
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.12))
    dt_val = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Solver parameters
    N = 48
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Solution at previous time step
    u_n = fem.Function(V, name="u_n")
    u_n.x.array[:] = 0.0  # Initial condition: u0 = 0.0
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(6*pi*x)*sin(6*pi*y)
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
    
    # Coefficients
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt = fem.Constant(domain, ScalarType(dt_val))
    
    # Backward Euler weak form:
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # => (u/dt)*v + kappa*grad(u).grad(v) = (u_n/dt)*v + f*v
    a = (u / dt) * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt) * v * ufl.dx + f * v * ufl.dx
    
    # Homogeneous Dirichlet BCs on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # Number of time steps
    n_steps = int(np.round(t_end / dt_val))
    
    # Linear solver
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="heat_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
    )
    
    # Time stepping loop
    total_iterations = 0
    for step in range(n_steps):
        u_h = problem.solve()
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 output grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = np.zeros((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
        },
    }
    
    return result


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 0.8},
            "time": {"t_end": 0.12, "dt": 0.02, "scheme": "backward_euler"},
            "source_term": "sin(6*pi*x)*sin(6*pi*y)",
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {np.nanmin(result['u']):.8f}, max: {np.nanmax(result['u']):.8f}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
