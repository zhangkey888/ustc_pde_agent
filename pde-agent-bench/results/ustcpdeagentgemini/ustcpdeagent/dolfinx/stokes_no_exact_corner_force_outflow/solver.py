import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution
    mesh_res = 64
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    gdim = domain.geometry.dim

    # Taylor-Hood elements
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = 0.1

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.as_vector((
        3 * ufl.exp(-50 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2)),
        3 * ufl.exp(-50 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
    ))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions
    fdim = domain.topology.dim - 1

    def boundary_x0(x):
        return np.isclose(x[0], 0.0)
    
    def boundary_y0(x):
        return np.isclose(x[1], 0.0)
    
    def boundary_y1(x):
        return np.isclose(x[1], 1.0)
    
    def all_boundaries(x):
        return boundary_x0(x) | boundary_y0(x) | boundary_y1(x)

    facets = mesh.locate_entities_boundary(domain, fdim, all_boundaries)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))

    bcs = [bc_u]

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    # Interpolate to grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    # Compute magnitude
    u_mag = np.linalg.norm(u_values, axis=1).reshape((ny, nx))

    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
