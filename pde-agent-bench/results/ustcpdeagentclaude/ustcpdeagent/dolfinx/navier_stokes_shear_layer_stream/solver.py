import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.18

    N = 128
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u, deg_p = 2, 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    # Manufactured solution
    u_ex = ufl.as_vector([
        6.0 * (1.0 - ufl.tanh(6.0 * (x[1] - 0.5))**2) * ufl.sin(pi * x[0]),
        -pi * ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.cos(pi * x[0]),
    ])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute forcing f from manufactured solution
    # f = u_ex · ∇u_ex - nu * ∇²u_ex + ∇p_ex
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = (ufl.grad(u_ex) * u_ex
         - nu_c * ufl.div(ufl.grad(u_ex))
         + ufl.grad(p_ex))

    # Weak form
    F = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f, v) * ufl.dx)

    # BCs: velocity = u_ex on entire boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0) - but match exact pressure value there
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        p_pin_func = fem.Function(Q)
        # p_ex(0,0) = cos(0)*cos(0) = 1
        p_pin_expr = fem.Expression(p_ex, Q.element.interpolation_points)
        p_pin_func.interpolate(p_pin_expr)
        bc_p = fem.dirichletbc(p_pin_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: interpolate exact solution as warm start
    u_init = fem.Function(V)
    u_init.interpolate(u_bc_expr)
    # Assign to w's velocity subspace
    # Easier: just set zero, Newton will converge
    w.x.array[:] = 0.0

    J = ufl.derivative(F, w)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_shear_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract solution
    u_sol = w.sub(0).collapse()

    # Sample on output grid
    out = case_spec["output"]["grid"]
    nx_g = out["nx"]; ny_g = out["ny"]
    bbox = out["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_g)
    ys = np.linspace(bbox[2], bbox[3], ny_g)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_g * ny_g)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    u_vals = np.full((nx_g * ny_g, gdim), np.nan)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[idx_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_g, nx_g)

    # Accuracy check: compute L2 error vs exact on mesh
    u_exact_fn = fem.Function(V)
    u_exact_fn.interpolate(u_bc_expr)
    err_form = fem.form(ufl.inner(u_sol - u_exact_fn, u_sol - u_exact_fn) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    err_global = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 velocity error: {err_global:.3e}")

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [int(problem.solver.getIterationNumber()) if hasattr(problem, "solver") else 0],
        },
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(case)
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Shape: {res['u'].shape}, min/max: {res['u'].min():.3f}/{res['u'].max():.3f}")
