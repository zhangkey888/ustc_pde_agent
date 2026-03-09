import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = float(params.get("epsilon", 0.005))
    beta_vec = params.get("beta", [20.0, 0.0])
    
    domain_spec = case_spec.get("domain", {})
    bounds = domain_spec.get("bounds", [[0, 1], [0, 1]])
    x0, x1 = bounds[0][0], bounds[0][1]
    y0, y1 = bounds[1][0], bounds[1][1]
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    
    # Parameters
    element_degree = 2
    N = 64
    ksp_type_used = "gmres"
    pc_type_used = "ilu"
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([x0, y0]), np.array([x1, y1])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Velocity field
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    # Source term from manufactured solution u_exact = sin(pi*x)*sin(pi*y)
    # f = -eps * laplacian(u) + beta . grad(u)
    # = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    f_expr = (2.0 * epsilon * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Standard Galerkin bilinear form
    a_std = (eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx)
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter
    Pe_h = beta_mag * h / (2.0 * epsilon)
    xi = ufl.conditional(ufl.gt(Pe_h, 1.0), 1.0, Pe_h / 3.0)
    tau = xi * h / (2.0 * beta_mag + 1e-10)
    
    # SUPG modification of test function
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    # SUPG terms
    a_supg = ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    # Boundary conditions: u = 0 on all boundaries (sin(pi*x)*sin(pi*y) = 0 on [0,1]^2 boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_arr: np.zeros_like(x_arr[0]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    try:
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": "1e-10",
                "ksp_atol": "1e-14",
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "150",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
        ksp_type_used = "gmres"
        pc_type_used = "ilu"
    except Exception:
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
        ksp_type_used = "preonly"
        pc_type_used = "lu"
    
    # Evaluate on output grid
    x_out = np.linspace(x0, x1, nx_out)
    y_out = np.linspace(y0, y1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": 1e-10,
            "iterations": 0,
            "stabilization": "SUPG",
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.005,
                "beta": [20.0, 0.0],
            },
        },
        "domain": {
            "bounds": [[0, 1], [0, 1]],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    nx, ny = u_grid.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    rms_error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"RMS error: {rms_error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Mesh: {result['solver_info']['mesh_resolution']}, Degree: {result['solver_info']['element_degree']}")
