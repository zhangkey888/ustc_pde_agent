import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # ── Parse case_spec ──
    pde = case_spec["pde"]
    eps = pde.get("diffusion", 0.02)
    beta_vec = pde.get("velocity", [-8.0, 4.0])
    
    out = case_spec["output"]
    grid_info = out["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    
    # ── Mesh parameters ──
    mesh_res = 44
    elem_degree = 2
    
    # ── Create mesh ──
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )
    
    # ── Function space ──
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # ── Exact solution u = exp(x)*sin(pi*y) ──
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(pi * x[1])
    
    # ── Source term ──
    e0 = ufl.exp(x[0])
    sin_py = ufl.sin(pi * x[1])
    cos_py = ufl.cos(pi * x[1])
    
    lap_u = e0 * sin_py * (1.0 - pi**2)
    grad_u_x = e0 * sin_py
    grad_u_y = pi * e0 * cos_py
    
    f_ufl = -eps * lap_u + beta_vec[0] * grad_u_x + beta_vec[1] * grad_u_y
    
    # ── Variational form with SUPG ──
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    beta = ufl.as_vector(beta_vec)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    
    tau_supg = h / (2.0 * beta_norm)
    
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a += ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    a += tau_supg * ufl.inner(beta, ufl.grad(u)) * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    
    L = f_ufl * v * ufl.dx
    L += tau_supg * f_ufl * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    
    # ── Boundary conditions ──
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # ── Solve with GMRES + ILU ──
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    # ── Sample solution onto output grid ──
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), 
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    u_values_global = np.zeros_like(u_values)
    MPI.COMM_WORLD.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # ── Build solver_info ──
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    time_spec = pde.get("time")
    if time_spec is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time as _time
    case_spec = {
        "pde": {"diffusion": 0.02, "velocity": [-8.0, 4.0], "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = _time.time()
    result = solve(case_spec)
    t1 = _time.time()
    print(f"Solve time: {t1-t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify accuracy
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(XX) * np.sin(np.pi * YY)
    err = result['u'] - u_exact
    max_err = np.max(np.abs(err))
    l2_err = np.sqrt(np.mean(err**2))
    print(f"Max pointwise error: {max_err:.6e}")
    print(f"L2-like error on grid: {l2_err:.6e}")
