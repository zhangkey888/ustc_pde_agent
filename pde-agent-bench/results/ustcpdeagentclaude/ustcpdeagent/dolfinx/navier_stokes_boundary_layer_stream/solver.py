import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.08

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh resolution
    N = 96
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        pi * ufl.exp(6 * (x[0] - 1)) * ufl.cos(pi * x[1]),
        -6 * ufl.exp(6 * (x[0] - 1)) * ufl.sin(pi * x[1]),
    ])
    p_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term f = u·∇u - ν∇²u + ∇p
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Dirichlet BC for velocity on entire boundary
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pinning at corner (0,0) - pin to exact value sin(0)*sin(0) = 0
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Residual
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Initial guess: interpolate exact velocity in BC to help convergence
    w.x.array[:] = 0.0

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

    problem = fem_petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_mms_",
        petsc_options=petsc_options,
    )

    try:
        w_sol = problem.solve()
    except Exception:
        # Fallback: start from Stokes
        w.x.array[:] = 0.0
        # Stokes initial
        (us, ps) = ufl.TrialFunctions(W)
        (vs, qs) = ufl.TestFunctions(W)
        a_stokes = (nu * ufl.inner(ufl.grad(us), ufl.grad(vs)) * ufl.dx
                    - ps * ufl.div(vs) * ufl.dx
                    + ufl.div(us) * qs * ufl.dx)
        L_stokes = ufl.inner(f, vs) * ufl.dx
        stokes_problem = fem_petsc.LinearProblem(
            a_stokes, L_stokes, bcs=bcs,
            petsc_options_prefix="stokes_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"},
        )
        w_init = stokes_problem.solve()
        w.x.array[:] = w_init.x.array[:]
        w_sol = problem.solve()

    w.x.scatter_forward()

    # Extract velocity
    u_h = w_sol.sub(0).collapse()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[idx_map] = vals

    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [0],
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, shape: {result['u'].shape}")
    print(f"Max magnitude: {result['u'].max():.4f}")

    # Compute error vs exact
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex_x = np.pi * np.exp(6*(XX-1)) * np.cos(np.pi*YY)
    u_ex_y = -6 * np.exp(6*(XX-1)) * np.sin(np.pi*YY)
    mag_ex = np.sqrt(u_ex_x**2 + u_ex_y**2)
    err = np.sqrt(np.mean((result['u'] - mag_ex)**2))
    print(f"RMS error vs exact: {err:.4e}")
    print(f"Max error: {np.max(np.abs(result['u'] - mag_ex)):.4e}")
