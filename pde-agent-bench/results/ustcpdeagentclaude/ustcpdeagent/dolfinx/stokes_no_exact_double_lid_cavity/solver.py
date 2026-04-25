import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.3

    N = 320
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # Boundary facets
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    facets_x1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facets_x0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))

    def make_bc(facets, value):
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        ub = fem.Function(V)
        ub.interpolate(lambda x: np.vstack([np.full(x.shape[1], value[0]),
                                            np.full(x.shape[1], value[1])]))
        return fem.dirichletbc(ub, dofs, W.sub(0))

    # Order: corners — apply x0 and y0 (zero) last so zero wins at corners?
    # Actually, more physical: lids take precedence on their edges; at corners, the
    # no-slip walls usually dominate. We'll apply no-slip last to override.
    bc_y1 = make_bc(facets_y1, (1.0, 0.0))
    bc_x1 = make_bc(facets_x1, (0.0, -0.8))
    bc_x0 = make_bc(facets_x0, (0.0, 0.0))
    bc_y0 = make_bc(facets_y0, (0.0, 0.0))

    bcs = [bc_y1, bc_x1, bc_x0, bc_y0]

    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    bcs.append(bc_p)

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_dlc_"
    )
    wh = problem.solve()

    try:
        its = problem.solver.getIterationNumber()
    except Exception:
        its = 1

    u_h = wh.sub(0).collapse()

    # Sample onto output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[idx_map] = vals

    mag = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(its),
        },
    }
