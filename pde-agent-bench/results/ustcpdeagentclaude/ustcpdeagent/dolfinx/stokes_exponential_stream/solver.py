import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 64
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u = 3
    deg_p = 2
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = 1.0
    x = ufl.SpatialCoordinate(msh)

    # Exact solution
    u_ex = ufl.as_vector([ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1]),
                          -ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])])
    p_ex = ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])

    # f = -nu*laplacian(u) + grad(p)
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Velocity BC (from exact solution)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin: enforce p = p_ex at (0,0)  -> p(0,0) = exp(0)*cos(0) = 1
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p_pin = fem.Function(Q)
    p_pin.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()
    iterations = problem.solver.getIterationNumber()

    u_h = w_h.sub(0).collapse()
    p_h = w_h.sub(1).collapse()

    # Compute L2 error for verification
    err_u = fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx)
    l2_err_u = np.sqrt(comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))
    print(f"L2 error u: {l2_err_u:.3e}")

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

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

    u_vals = np.zeros((pts.shape[0], gdim))
    if points_on_proc:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[idx_map] = vals

    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(max(iterations, 1)),
        }
    }


if __name__ == "__main__":
    import time
    case = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(case)
    print(f"time: {time.time()-t0:.2f}s, shape: {res['u'].shape}")
