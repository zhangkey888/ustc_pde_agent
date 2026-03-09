import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Default parameters from problem description
    epsilon = 0.02
    beta_vec = [-8.0, 4.0]
    
    # Parse case_spec if provided
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        epsilon = params.get("epsilon", epsilon)
        beta_vec = params.get("beta", beta_vec)
    
    # Adaptive mesh refinement loop
    resolutions = [48, 80, 128]
    prev_norm = None
    final_result = None
    
    for N in resolutions:
        result = _solve_at_resolution(N, epsilon, beta_vec)
        curr_norm = result["norm"]
        
        if prev_norm is not None:
            rel_change = abs(curr_norm - prev_norm) / (abs(curr_norm) + 1e-15)
            if rel_change < 0.005:
                return result["output"]
        
        prev_norm = curr_norm
        final_result = result["output"]
    
    return final_result


def _solve_at_resolution(N, epsilon, beta_vec):
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    pi = ufl.pi
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(pi * x[1])
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    f_expr = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # Standard Galerkin
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = ufl.inner(f_expr, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    
    tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    r_test = ufl.dot(beta, ufl.grad(v))
    
    a_supg = tau * r_test * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L_supg = tau * r_test * f_expr * ufl.dx
    
    a = a_std + a_supg
    L = L_std + L_supg
    
    # Boundary conditions
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": "1e-10",
                "ksp_max_it": "2000",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
        ksp_type = "gmres"
        pc_type = "ilu"
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
        ksp_type = "preonly"
        pc_type = "lu"
    
    l2_norm_val = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
        op=MPI.SUM
    ))
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
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
    
    output = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
        }
    }
    
    return {"norm": l2_norm_val, "output": output}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.3f}s")
    
    nx, ny = 50, 50
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(XX) * np.sin(np.pi * YY)
    
    u_num = result["u"]
    err = np.sqrt(np.mean((u_num - u_exact)**2))
    max_err = np.max(np.abs(u_num - u_exact))
    print(f"L2 error (grid): {err:.6e}")
    print(f"Max error (grid): {max_err:.6e}")
    print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
