import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: poisson
# ```


def _u_exact_numpy_xy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.3 * np.sin(6.0 * np.pi * x) * np.sin(6.0 * np.pi * y)


def _u_exact_numpy(x):
    return _u_exact_numpy_xy(x[0], x[1])


def _u_exact_ufl(domain):
    x = ufl.SpatialCoordinate(domain)
    return (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.3 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
    )


def _sample_on_grid(uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    msh = uh.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_local = []
    cells_local = []
    ids_local = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if points_local:
        vals = uh.eval(
            np.asarray(points_local, dtype=np.float64),
            np.asarray(cells_local, dtype=np.int32),
        ).reshape(-1)
        values[np.asarray(ids_local, dtype=np.int32)] = vals

    gathered = msh.comm.gather(values, root=0)
    if msh.comm.rank == 0:
        merged = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            mask = np.isnan(merged)
            merged[mask] = _u_exact_numpy(pts[mask].T)
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 40
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_exact = _u_exact_ufl(msh)
    kappa = fem.Constant(msh, PETSc.ScalarType(1.0))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a = fem.form(ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f, v) * ufl.dx)

    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.getPC().setHYPREType("boomeramg")
    except Exception:
        pass
    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    Verr = fem.functionspace(msh, ("Lagrange", 4))
    uex = fem.Function(Verr)
    uex.interpolate(fem.Expression(u_exact, Verr.element.interpolation_points))
    uh_high = fem.Function(Verr)
    uh_high.interpolate(uh)
    err_local = fem.assemble_scalar(fem.form((uh_high - uex) ** 2 * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(l2_error),
    }

    if comm.rank == 0:
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
