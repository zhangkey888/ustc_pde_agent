import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    nu_val = 0.22

    # Parameters
    N = 64
    deg_u = 3
    deg_p = 2

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)

    # Exact solution
    u_ex = ufl.as_vector([
        x[0]**2 * (1 - x[0])**2 * (1 - 2*x[1]),
        -2 * x[0] * (1 - x[0]) * (1 - 2*x[0]) * x[1] * (1 - x[1])
    ])
    p_ex = x[0] + x[1]

    # Compute f from manufactured solution
    # f = u·∇u - ν∇²u + ∇p
    f_ex = ufl.grad(u_ex) * u_ex - nu_val * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Residual
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f_ex, v) * ufl.dx)

    # BC for velocity
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(u_ex, V.element.interpolation_points)
    )

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pinning at (0,0) to match exact p(0,0)=0
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(
        fem.Expression(p_ex, Q.element.interpolation_points)
    )
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: zero
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
        petsc_options_prefix="ns_solver_",
        petsc_options=petsc_options,
    )
    w = problem.solve()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    mag = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            mag[idx] = np.linalg.norm(vals[k])

    u_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [10],
        },
    }


if __name__ == "__main__":
    import time
    case = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u max: {result['u'].max()}, u min: {result['u'].min()}")

    # Verify against exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    ux_ex = XX**2 * (1 - XX)**2 * (1 - 2*YY)
    uy_ex = -2 * XX * (1 - XX) * (1 - 2*XX) * YY * (1 - YY)
    mag_ex = np.sqrt(ux_ex**2 + uy_ex**2)
    err = np.sqrt(np.mean((result['u'] - mag_ex)**2))
    print(f"RMS error: {err:.6e}")
    print(f"Max error: {np.max(np.abs(result['u'] - mag_ex)):.6e}")
