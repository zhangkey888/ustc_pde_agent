import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow on a unit square channel using Taylor-Hood mixed FEM."""
    t_start = time.time()

    # Extract PDE parameters
    pde = case_spec["pde"]
    nu = float(pde["nu"])
    f_vals = pde["source"]

    # Output grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]

    # Mesh resolution and element degrees (P3/P2 Taylor-Hood for high accuracy)
    N = 96
    degree_u = 3
    degree_p = 2

    # Create mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood mixed element: P3 velocity, P2 pressure
    vel_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix.ufl.mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Variational form: -nu * laplacian(u) + grad(p) = f, div(u) = 0
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(msh, PETSc.ScalarType((float(f_vals[0]), float(f_vals[1]))))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1

    # Inlet: x = 0, u = [4*y*(1-y), 0]
    inlet_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_inlet = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    u_inlet = fem.Function(V)
    u_inlet.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    bc_inlet = fem.dirichletbc(u_inlet, dofs_inlet, W.sub(0))

    # Bottom wall: y = 0, u = [0, 0]
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0))

    # Top wall: y = 1, u = [0, 0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.x.array[:] = 0.0
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))

    # Pressure pin at origin corner (fix gauge pressure for uniqueness)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_inlet, bc_bottom, bc_top, bc_p]

    # Solve with KLU direct solver (fast and reliable for Stokes saddle-point)
    rtol = 1e-10
    problem = petsc.LinearProblem(a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "klu",
        },
        petsc_options_prefix="stokes_")

    w_h = problem.solve()

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())

    w_h.x.scatter_forward()
    u_h = w_h.sub(0).collapse()

    # Sample velocity onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])  # (3, N)

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals

    # Compute velocity magnitude
    u_grid = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    # Build solver info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": iterations,
    }

    # Check time info (this is a steady problem)
    pde_info = case_spec.get("pde", {})
    time_info = pde_info.get("time", None)
    if time_info is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    t_end = time.time()

    return {"u": u_grid, "solver_info": solver_info}
