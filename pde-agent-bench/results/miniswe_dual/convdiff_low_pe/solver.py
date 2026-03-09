import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    eps_val = 0.2
    beta_vec = [1.0, 0.5]
    
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        eps_val = params.get("epsilon", eps_val)
        beta_val = params.get("beta", beta_vec)
        if beta_val is not None:
            beta_vec = beta_val

    comm = MPI.COMM_WORLD
    nx_out, ny_out = 50, 50
    
    N = 48
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    eps_c = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))
    
    f_expr = (eps_val * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin bilinear form and RHS
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_local = beta_mag * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_local) - 1.0 / Pe_local)
    
    # SUPG test function modification: tau * (beta . grad(v))
    # Strong form operator applied to u: -eps * div(grad(u)) + beta . grad(u)
    # Split into bilinear part (operator on u) and linear part (f)
    # Bilinear: tau * (-eps * div(grad(u)) + beta . grad(u)) * (beta . grad(v))
    # Linear:   tau * f * (beta . grad(v))
    
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    # Bilinear SUPG: operator applied to trial function u
    a += (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * supg_test * ufl.dx
    
    # Linear SUPG: source term
    L += f_expr * supg_test * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="cdiff_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
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
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        },
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    info = result["solver_info"]
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"Mesh: {info['mesh_resolution']}, Degree: {info['element_degree']}")
    print(f"RMS Error: {error:.6e}")
    print(f"Max Error: {np.max(np.abs(u_grid - u_exact)):.6e}")
    print(f"Wall time: {elapsed:.3f}s")
