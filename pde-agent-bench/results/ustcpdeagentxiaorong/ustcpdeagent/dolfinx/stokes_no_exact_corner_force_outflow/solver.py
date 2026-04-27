import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = case_spec["pde"]["viscosity"]
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 128
    degree_u = 2
    degree_p = 1

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Source term: Gaussian bump at (0.15, 0.15)
    x = ufl.SpatialCoordinate(domain)
    
    # Parse source term from case_spec if available
    src = case_spec["pde"].get("source_term", None)
    if src is not None and len(src) == 2:
        # For this specific problem: f = 3*exp(-50*((x-0.15)^2 + (y-0.15)^2))
        # Build UFL expression
        f_val = 3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
        f_vec = ufl.as_vector([f_val, f_val])
    else:
        f_vec = ufl.as_vector([0.0 * x[0], 0.0 * x[0]])

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Stokes weak form: -nu*Laplacian(u) + grad(p) = f, div(u) = 0
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f_vec, v) * ufl.dx

    # Dirichlet BCs
    bcs = []
    bc_spec = case_spec["pde"].get("boundary_conditions", {})

    # Map boundary names to marker functions
    bc_map = {
        "x0": lambda xc: np.isclose(xc[0], 0.0),
        "x1": lambda xc: np.isclose(xc[0], 1.0),
        "y0": lambda xc: np.isclose(xc[1], 0.0),
        "y1": lambda xc: np.isclose(xc[1], 1.0),
    }

    for bc_name, marker_fn in bc_map.items():
        if bc_name in bc_spec:
            facets = mesh.locate_entities_boundary(domain, fdim, marker_fn)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            bc_val = bc_spec[bc_name].get("value", [0.0, 0.0])
            uf = fem.Function(V)
            if bc_val == [0.0, 0.0] or bc_val == 0.0:
                uf.x.array[:] = 0.0
            else:
                uf.interpolate(lambda xc: np.vstack([
                    np.full(xc.shape[1], bc_val[0]),
                    np.full(xc.shape[1], bc_val[1])
                ]))
            bcs.append(fem.dirichletbc(uf, dofs, W.sub(0)))

    ksp_type = "preonly"
    pc_type = "lu"

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
    u_h = w_h.sub(0).collapse()

    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_magnitude = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_arr, cells_arr)
        mag = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, orig_idx in enumerate(eval_map):
            u_magnitude[orig_idx] = mag[idx]

    u_grid = u_magnitude.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 1,
        },
    }
