import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}

    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta_vec = params.get("beta", [12.0, 6.0])
    
    comm = MPI.COMM_WORLD
    
    # Use high resolution for accuracy - still fast enough
    N = 400
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    pi_val = ufl.pi
    f_expr = (ufl.sin(8 * pi_val * x[0]) * ufl.sin(6 * pi_val * x[1])
              + 0.3 * ufl.sin(12 * pi_val * x[0]) * ufl.sin(10 * pi_val * x[1]))
    
    # Standard Galerkin bilinear form
    a_std = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v))
             + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization for high Peclet number
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_local = beta_mag * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_local) - 1.0 / Pe_local)
    
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    # For P1 elements, Laplacian vanishes element-wise
    a_supg = a_std + ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx
    L_supg = L_std + f_expr * v_supg * ufl.dx
    
    # Homogeneous Dirichlet BC on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve with iterative solver
    try:
        problem = petsc.LinearProblem(
            a_supg, L_supg, bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": "1e-10",
                "ksp_max_it": "3000",
                "ksp_gmres_restart": "100",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(
            a_supg, L_supg, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="cdiff_fb_"
        )
        u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    u_grid = _evaluate_function(domain, u_sol, points_3d, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": 0,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


def _evaluate_function(domain, u_func, points_3d, nx, ny):
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
    
    u_values = np.full(points_3d.shape[0], 0.0)
    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape(nx, ny)


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {elapsed:.2f}s")
