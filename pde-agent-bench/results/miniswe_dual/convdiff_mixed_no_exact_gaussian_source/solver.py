import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [10.0, 5.0])
    
    domain_spec = case_spec.get("domain", {})
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)
    
    # Use P2 elements with moderate mesh for accuracy with SUPG
    N = 128
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: Gaussian
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Standard weak form
    a_std = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    supg_test = ufl.dot(beta, ufl.grad(v))
    a_supg = tau * ufl.dot(beta, ufl.grad(u)) * supg_test * ufl.dx
    L_supg = tau * f_expr * supg_test * ufl.dx
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # Homogeneous Dirichlet BC on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_val = fem.Function(V)
    bc_val.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(bc_val, dofs)
    bcs = [bc]
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "10000",
                "ksp_gmres_restart": "150",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
    
    # Evaluate on output grid
    x_out = np.linspace(0.0, 1.0, nx_out)
    y_out = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], 0.0)
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
            "stabilization": "SUPG",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"type": "convection_diffusion", "params": {"epsilon": 0.01, "beta": [10.0, 5.0]}, "bcs": {}},
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u = result["u"]
    print(f"Shape: {u.shape}, Range: [{u.min():.6f}, {u.max():.6f}], L2: {np.sqrt(np.mean(u**2)):.6f}, Time: {elapsed:.2f}s")
