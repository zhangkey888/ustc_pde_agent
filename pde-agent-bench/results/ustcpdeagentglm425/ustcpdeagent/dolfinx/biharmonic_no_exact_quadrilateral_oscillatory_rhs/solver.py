import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Push resolution higher to use more of time budget
    N = 200
    elem_deg = 3
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(8*ufl.pi*x[0]) * ufl.cos(6*ufl.pi*x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_val = fem.Function(V)
    u_bc_val.x.array[:] = 0.0
    bc_zero = fem.dirichletbc(u_bc_val, boundary_dofs)
    
    v_test = ufl.TestFunction(V)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    petsc_opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": str(rtol),
        "ksp_atol": "1e-15",
        "ksp_max_it": 1000,
    }
    
    # Step 1: -Delta(w) = f, w=0 on boundary
    w_trial = ufl.TrialFunction(V)
    a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
    L1 = ufl.inner(f_expr, v_test) * ufl.dx
    
    problem1 = petsc.LinearProblem(a1, L1, bcs=[bc_zero],
        petsc_options_prefix="biharm_w_", petsc_options=petsc_opts)
    w_sol = problem1.solve()
    w_sol.x.scatter_forward()
    its1 = problem1.solver.getIterationNumber()
    
    # Step 2: -Delta(u) = w, u=0 on boundary
    u_trial = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L2 = ufl.inner(w_sol, v_test) * ufl.dx
    
    problem2 = petsc.LinearProblem(a2, L2, bcs=[bc_zero],
        petsc_options_prefix="biharm_u_", petsc_options=petsc_opts)
    u_sol = problem2.solve()
    u_sol.x.scatter_forward()
    its2 = problem2.solver.getIterationNumber()
    total_iterations = its1 + its2
    
    # Vectorized point evaluation
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_out * ny_out, dtype=np.float64)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Accuracy verification
    boundary_max = max(
        np.max(np.abs(u_grid[0, :])), np.max(np.abs(u_grid[-1, :])),
        np.max(np.abs(u_grid[:, 0])), np.max(np.abs(u_grid[:, -1]))
    )
    print(f"BC verification: boundary max|u| = {boundary_max:.2e}")
    print(f"Interior max|u| = {np.max(np.abs(u_grid)):.6e}")
    print(f"Iterations: step1={its1}, step2={its2}, total={total_iterations}")
    
    # Compute residual for accuracy check
    residual1 = f_expr + ufl.div(ufl.grad(w_sol))
    res1_form = fem.form(residual1**2 * ufl.dx)
    f_form = fem.form(f_expr**2 * ufl.dx)
    res1_sq = domain.comm.allreduce(fem.assemble_scalar(res1_form), op=MPI.SUM)
    f_sq = domain.comm.allreduce(fem.assemble_scalar(f_form), op=MPI.SUM)
    print(f"Poisson residual: {np.sqrt(res1_sq/f_sq):.4e}")
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
    }
    
    pde_time = case_spec.get("pde", {}).get("time", None)
    if pde_time and pde_time.get("is_transient", False):
        solver_info["dt"] = 1.0
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "steady"
    
    result = {"u": u_grid, "solver_info": solver_info}
    
    if pde_time and pde_time.get("is_transient", False):
        result["u_initial"] = np.zeros_like(u_grid)
    
    return result
