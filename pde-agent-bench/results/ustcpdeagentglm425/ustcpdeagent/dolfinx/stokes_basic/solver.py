import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    nu = 1.0
    pi = np.pi

    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]

    N = 256

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Source term from manufactured solution
    # u_ex = [pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y)]
    # p_ex = cos(pi*x)*cos(pi*y)
    # f = -nu*laplacian(u) + grad(p)
    x = ufl.SpatialCoordinate(msh)
    f_val = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) * (2 * pi**2 - 1),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) * (2 * pi**2 + 1)
    ])

    # Bilinear form
    a_form = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              - p * ufl.div(v) * ufl.dx
              + q * ufl.div(u) * ufl.dx)
    L_form = ufl.inner(f_val, v) * ufl.dx

    # Exact solution for BCs
    u_ex_expr = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_ex_expr = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Velocity BC on entire boundary (topological approach)
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at corner (0,0) - exact pressure = cos(0)*cos(0) = 1.0
    p_dofs_Q_local = fem.locate_dofs_geometrical(Q, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bcs = [bc_u]
    if len(p_dofs_Q_local) > 0:
        p_dofs_W_local = np.array([Q_to_W[dof] for dof in p_dofs_Q_local], dtype=np.int32)
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 1.0
        bc_p = fem.dirichletbc(p0_func, [p_dofs_W_local, p_dofs_Q_local], W.sub(1))
        bcs.append(bc_p)

    # Solve using MUMPS with nullspace detection
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_7": 6,
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    w_h.x.scatter_forward()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Extract velocity and pressure
    u_h = w_h.sub(0).collapse()
    p_h = w_h.sub(1).collapse()

    # Accuracy verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))
    p_ex_func = fem.Function(Q)
    p_ex_func.interpolate(fem.Expression(p_ex_expr, Q.element.interpolation_points))

    err_u_sq = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx))
    err_u_L2 = np.sqrt(comm.allreduce(err_u_sq, op=MPI.SUM))

    err_p_sq = fem.assemble_scalar(fem.form((p_h - p_ex_func)**2 * ufl.dx))
    err_p_L2 = np.sqrt(comm.allreduce(err_p_sq, op=MPI.SUM))

    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    point_indices = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            point_indices.append(i)

    u_grid_local = np.zeros((ny_out, nx_out))

    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mag = np.linalg.norm(u_vals, axis=1)
        for idx, val in zip(point_indices, mag):
            iy = idx // nx_out
            ix = idx % nx_out
            u_grid_local[iy, ix] = val

    u_grid = np.zeros_like(u_grid_local)
    comm.Allreduce(u_grid_local, u_grid, op=MPI.SUM)

    # Pointwise error on grid for verification
    u_ex_grid = np.sqrt(
        (pi * np.cos(pi * YY) * np.sin(pi * XX))**2 +
        (pi * np.cos(pi * XX) * np.sin(pi * YY))**2
    )
    pointwise_err = np.max(np.abs(u_grid - u_ex_grid))

    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        print(f"Mesh: {N}x{N}, Vel L2: {err_u_L2:.6e}, Pres L2: {err_p_L2:.6e}")
        print(f"Pointwise max err: {pointwise_err:.6e}, Time: {elapsed:.2f}s, Iters: {iterations}")

    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    result = solve(case_spec)
    print(f"Shape: {result['u'].shape}, Max: {np.max(result['u']):.6e}")
