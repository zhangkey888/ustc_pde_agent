import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid info
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Parameters
    nu_val = 1.0
    N = 128  # mesh resolution
    element_degree_vel = 2
    element_degree_pres = 1

    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed element (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), element_degree_vel, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), element_degree_pres)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # Manufactured solution
    u_exact = ufl.as_vector([
        pi * ufl.exp(x[0]) * ufl.cos(pi * x[1]),
        -ufl.exp(x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(pi * x[1])

    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f_expr = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Bilinear form for Stokes
    a = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)

    # Linear form
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1

    # All boundary facets (topological approach)
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pinning at origin to exact value
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_pin = fem.Function(Q)
    p_pin.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Solve with MUMPS direct solver (handles saddle-point systems)
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

    # Sample velocity on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    # Build bounding box tree and find cells
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

    u_values = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree_vel,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
