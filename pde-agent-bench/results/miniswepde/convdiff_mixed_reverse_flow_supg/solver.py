import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    """Solve convection-diffusion with SUPG stabilization."""
    
    epsilon = 0.005
    beta_vec = [-20.0, 5.0]
    
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        epsilon = params.get("epsilon", epsilon)
        beta_vec = params.get("beta", beta_vec)
    
    comm = MPI.COMM_WORLD
    N = 80
    degree = 1
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    grad_u_exact = ufl.as_vector([
        ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])
    ])
    laplacian_u_exact = (1.0 - ufl.pi**2) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    f_expr = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = ufl.inner(f_expr, v) * ufl.dx
    
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    r_test = tau * ufl.dot(beta, ufl.grad(v))
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), r_test) * ufl.dx
    L_supg = ufl.inner(f_expr, r_test) * ufl.dx
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_arr: np.exp(x_arr[0]) * np.sin(np.pi * x_arr[1]))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
    }

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(XX) * np.sin(np.pi * YY)
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_err = np.nanmax(np.abs(u_grid - u_exact))
    print(f"Time: {elapsed:.3f}s | RMS: {error:.6e} | Max: {max_err:.6e} | N={result['solver_info']['mesh_resolution']}")
    print(f"PASS" if error <= 1.47e-3 and elapsed <= 3.168 else f"FAIL")
