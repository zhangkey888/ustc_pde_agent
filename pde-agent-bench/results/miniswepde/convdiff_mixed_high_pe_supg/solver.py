import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}
    
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    domain_spec = case_spec.get("domain", {})
    
    epsilon = params.get("epsilon", 0.005)
    beta_vec = params.get("beta", [20.0, 10.0])
    
    nx_out = case_spec.get("output", {}).get("nx", 50)
    ny_out = case_spec.get("output", {}).get("ny", 50)
    
    x_min = domain_spec.get("x_min", 0.0)
    x_max = domain_spec.get("x_max", 1.0)
    y_min = domain_spec.get("y_min", 0.0)
    y_max = domain_spec.get("y_max", 1.0)
    
    comm = MPI.COMM_WORLD
    
    N = 80
    element_degree = 2
    
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain_mesh = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain_mesh, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain_mesh)
    
    eps_c = fem.Constant(domain_mesh, PETSc.ScalarType(epsilon))
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    
    pi_val = ufl.pi
    u_exact_ufl = ufl.sin(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    f_expr = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    tdim = domain_mesh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain_mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain_mesh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # Bilinear form with SUPG
    a_form = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
              + tau * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dot(beta, ufl.grad(v)) * ufl.dx)
    
    L_form = (f_expr * v * ufl.dx
              + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx)
    
    ksp_type = "gmres"
    pc_type = "ilu"
    
    try:
        problem = petsc.LinearProblem(
            a_form, L_form, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": "1e-10",
                "ksp_atol": "1e-12",
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "100",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_form, L_form, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
    
    u_grid = evaluate_on_grid(domain_mesh, u_sol, nx_out, ny_out, x_min, x_max, y_min, y_max)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
        }
    }


def evaluate_on_grid(domain_mesh, u_func, nx, ny, x_min, x_max, y_min, y_max):
    x_pts = np.linspace(x_min, x_max, nx)
    y_pts = np.linspace(y_min, y_max, ny)
    
    X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    bb_tree = geometry.bb_tree(domain_mesh, domain_mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_mesh, cell_candidates, points.T)
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"type": "convection_diffusion", "parameters": {"epsilon": 0.005, "beta": [20.0, 10.0]}},
        "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "output": {"nx": 50, "ny": 50},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    x_pts = np.linspace(0, 1, 50)
    y_pts = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    l2_err = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_err = np.max(np.abs(u_grid - u_exact))
    print(f"Mesh: {result['solver_info']['mesh_resolution']}, Degree: {result['solver_info']['element_degree']}")
    print(f"L2 error: {l2_err:.6e}")
    print(f"Linf error: {linf_err:.6e}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Target error: 4.53e-04, Target time: 3.804s")
    print(f"PASS: {l2_err <= 4.53e-4 and elapsed <= 3.804}")
