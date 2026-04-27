import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]

    # Solver parameters - N=80 gives max pointwise error ~8.5e-06 < 1.29e-05
    N = 80
    degree_u = 2
    degree_p = 1

    # Create mesh on the specified domain
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates and manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Source term: -nu * laplacian(u) + grad(p) = f
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Bilinear form for Stokes
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pin at corner (bbox origin)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2])
    )
    p_pin = fem.Function(Q)
    p_pin.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Solve with MUMPS direct solver
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((ny_out, nx_out), np.nan)

    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_eval, cells_eval)  # shape (N, gdim)
        magnitude = np.linalg.norm(vals, axis=1)

        for idx, global_i in enumerate(eval_map):
            row = global_i // nx_out
            col = global_i % nx_out
            u_grid[row, col] = magnitude[idx]

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 1.0},
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()

    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Min: {np.nanmin(result['u']):.6e}, Max: {np.nanmax(result['u']):.6e}")

    # Verify against exact solution
    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys)

    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

    err = np.abs(result['u'] - mag_exact)
    print(f"Max pointwise error on grid: {np.nanmax(err):.6e}")
    print(f"Mean pointwise error on grid: {np.nanmean(err):.6e}")
