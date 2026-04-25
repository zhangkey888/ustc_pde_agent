import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]

    nu_val = 0.2
    N = 96
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
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
        pi * ufl.cos(pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_ex = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute f = (u_ex · grad) u_ex - nu Δu_ex + grad p_ex
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    # Residual
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f, v) * ufl.dx)

    # BCs: velocity = u_ex on boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pin pressure at (0,0) to exact value
    p_bc = fem.Function(Q)
    p_bc.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    bc_p = fem.dirichletbc(p_bc, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: interpolate exact velocity (helps Newton)
    w.x.array[:] = 0.0
    # Set initial guess for velocity part
    w_u = w.sub(0)
    # Interpolate u_ex into velocity subspace via helper
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    # Copy u_init into w.sub(0)
    # Use a helper: assign through a fresh mixed function by interpolation
    # Simpler: solve Stokes first for an initial guess

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
    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_sol = w.sub(0).collapse()
    p_sol = w.sub(1).collapse()

    # Compute L2 error for verification
    err_u = fem.form(ufl.inner(u - u_ex, u - u_ex) * ufl.dx)
    err_val = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))
    print(f"L2 velocity error: {err_val:.3e}")

    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    cells = []
    points_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "nonlinear_iterations": [problem.solver.getIterationNumber() if hasattr(problem, 'solver') else 0],
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Shape: {result['u'].shape}, max: {result['u'].max():.4f}")
