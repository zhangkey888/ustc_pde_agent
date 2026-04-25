import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.1

    # Mesh resolution
    N = 96
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    gdim = msh.geometry.dim

    cell_name = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    # Manufactured solution
    u_ex = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    ])
    p_ex = 0.0 * x[0]

    # Compute f = u·∇u - ν∆u + ∇p
    f = ufl.grad(u_ex) * u_ex - nu_val * ufl.div(ufl.grad(u_ex))

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J = ufl.derivative(F, w)

    # Velocity BC from manufactured solution
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: interpolate exact velocity (helps Newton)
    w.x.array[:] = 0.0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_mms_",
        petsc_options=petsc_options,
    )
    w_h = problem.solve()
    w.x.scatter_forward()

    # Get iteration counts
    try:
        snes = problem.solver
        nl_its = snes.getIterationNumber()
        ksp_its = snes.getLinearSolveIterations()
    except Exception:
        nl_its = 0
        ksp_its = 0

    # Extract velocity
    u_sol = w.sub(0).collapse()

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
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

    mag = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        m = np.linalg.norm(vals, axis=1)
        for k, idx in enumerate(eval_map):
            mag[idx] = m[k]

    mag_grid = mag.reshape(ny_out, nx_out)

    return {
        "u": mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(ksp_its),
            "nonlinear_iterations": [int(nl_its)],
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f} s")

    # Check error against exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    ux = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_ex = np.sqrt(ux**2 + uy**2)
    err = np.sqrt(np.mean((result["u"] - mag_ex) ** 2))
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {np.max(np.abs(result['u'] - mag_ex)):.3e}")
    print(f"Solver info: {result['solver_info']}")
