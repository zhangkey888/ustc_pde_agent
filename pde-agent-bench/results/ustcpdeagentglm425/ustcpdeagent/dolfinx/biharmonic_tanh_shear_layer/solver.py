import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    domain_info = pde["domain"]
    xmin, xmax = domain_info["x_range"]
    ymin, ymax = domain_info["y_range"]
    
    output_spec = case_spec["output"]
    nx_out = output_spec["grid"]["nx"]
    ny_out = output_spec["grid"]["ny"]
    bbox = output_spec["grid"]["bbox"]
    out_xmin, out_xmax, out_ymin, out_ymax = bbox

    N = 94
    elem_deg = 4
    rtol_val = 1e-12

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Manufactured solution: u = tanh(6*(y-0.5))*sin(pi*x)
    u_exact = ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(pi * x[0])
    lap_u_exact = ufl.div(ufl.grad(u_exact))
    f_expr = ufl.div(ufl.grad(lap_u_exact))  # f = Delta^2 u

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Step 1: Solve Delta phi = f with phi = Delta u_exact on boundary
    # FEM standard: -Delta phi = -f => a(phi,v) = -<f,v>
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    L1 = -ufl.inner(f_expr, v) * ufl.dx

    phi_bc_func = fem.Function(V)
    phi_bc_func.interpolate(fem.Expression(lap_u_exact, V.element.interpolation_points))
    bc1 = fem.dirichletbc(phi_bc_func, boundary_dofs)

    phi_h = fem.Function(V)
    problem1 = petsc.LinearProblem(
        a1, L1, u=phi_h, bcs=[bc1],
        petsc_options_prefix="biharm1_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": str(rtol_val)},
    )
    phi_h = problem1.solve()
    phi_h.x.scatter_forward()
    its1 = problem1.solver.getIterationNumber()

    # Step 2: Solve Delta u = phi_h with u = g on boundary
    # FEM standard: -Delta u = -phi_h => a(u,v) = -<phi_h,v>
    u2 = ufl.TrialFunction(V)
    v2 = ufl.TestFunction(V)
    a2 = ufl.inner(ufl.grad(u2), ufl.grad(v2)) * ufl.dx
    L2 = -ufl.inner(phi_h, v2) * ufl.dx

    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc2 = fem.dirichletbc(g_func, boundary_dofs)

    uh = fem.Function(V)
    problem2 = petsc.LinearProblem(
        a2, L2, u=uh, bcs=[bc2],
        petsc_options_prefix="biharm2_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": str(rtol_val)},
    )
    uh = problem2.solve()
    uh.x.scatter_forward()
    its2 = problem2.solver.getIterationNumber()

    # ---- Accuracy verification ----
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    error_form = fem.form(ufl.inner(uh - u_exact_func, uh - u_exact_func) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    if MPI.COMM_WORLD.rank == 0:
        print(f"[Accuracy] L2 error = {error_global:.6e}")

    # ---- Sample solution on output grid ----
    xs = np.linspace(out_xmin, out_xmax, nx_out)
    ys = np.linspace(out_ymin, out_ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    local_idx = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            local_idx.append(i)

    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[local_idx] = vals.flatten()

    if domain.comm.size > 1:
        gathered = domain.comm.allgather(u_values)
        u_global = np.full_like(u_values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            u_global[mask] = arr[mask]
        u_grid = u_global.reshape(ny_out, nx_out)
    else:
        u_grid = u_values.reshape(ny_out, nx_out)

    total_iters = int(its1 + its2)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[Solver] its1={its1}, its2={its2}, total={total_iters}")

    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol_val,
        "iterations": total_iters,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "name": "biharmonic_tanh_shear_layer",
            "type": "elliptic",
            "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        },
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    t1 = time.perf_counter()
    print(f"Wall time: {t1 - t0:.3f} s")
    print(f"Output shape: {result['u'].shape}")
    print(f"u min/max: {np.nanmin(result['u']):.6f} / {np.nanmax(result['u']):.6f}")
