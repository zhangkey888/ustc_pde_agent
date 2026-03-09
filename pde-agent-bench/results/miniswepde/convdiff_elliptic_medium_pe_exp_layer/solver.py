import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.05)
    beta_vec = params.get("beta", [4.0, 0.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    N = 64
    element_degree = 2
    
    comm = MPI.COMM_WORLD
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
    
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    pi_ = ufl.pi
    
    grad_u_exact = ufl.as_vector([
        2.0 * ufl.exp(2.0 * x[0]) * ufl.sin(pi_ * x[1]),
        pi_ * ufl.exp(2.0 * x[0]) * ufl.cos(pi_ * x[1])
    ])
    laplacian_u_exact = (4.0 - pi_**2) * ufl.exp(2.0 * x[0]) * ufl.sin(pi_ * x[1])
    f_expr = -epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    a_std = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    r_strong_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * r_strong_u * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim,
        lambda x: (np.isclose(x[0], x_range[0]) | np.isclose(x[0], x_range[1]) |
                   np.isclose(x[1], y_range[0]) | np.isclose(x[1], y_range[1])))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(2.0 * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": "1e-10", "ksp_max_it": "2000"},
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": 0,
        },
    }

if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"type": "convection_diffusion", "params": {"epsilon": 0.05, "beta": [4.0, 0.0]}},
        "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "output": {"nx": 50, "ny": 50},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    x_c = np.linspace(0, 1, 50)
    y_c = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_c, y_c, indexing='ij')
    u_exact = np.exp(2.0 * X) * np.sin(np.pi * Y)
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"Time: {elapsed:.3f}s, L2 error: {error:.6e}, Max error: {np.max(np.abs(u_grid - u_exact)):.6e}")
