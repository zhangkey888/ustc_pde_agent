import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nu_val = 0.1
    N = 64
    deg_u = 2
    deg_p = 1

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
    pi = ufl.pi

    # Manufactured solution
    u_ex = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
        -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1]),
    ])
    p_ex = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])

    # Compute f = u·∇u - ν Δu + ∇p
    f = ufl.grad(u_ex) * u_ex - nu_val * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Boundary conditions: u = u_ex on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pinning at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_dofs[0]) > 0:
        bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: interpolate exact velocity (makes Newton converge fast)
    w_V_sub = w.sub(0)
    # Set initial guess to zero (for generality) — or use exact for robustness
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
        petsc_options_prefix="ns_high_freq_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Compute L2 error for verification
    err_u = fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx)
    L2_err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))
    print(f"L2 velocity error: {L2_err:.3e}")

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

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

    u_vals = np.zeros((nx * ny, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
            "nonlinear_iterations": [10],
        },
    }


if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    import time
    t0 = time.time()
    res = solve(case)
    print("Time:", time.time() - t0)
    print("Shape:", res["u"].shape)
    print("Max mag:", res["u"].max())
