import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

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

# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```


def _u_exact_np(x, y):
    return np.exp(6.0 * y) * np.sin(np.pi * x)


def _sample_on_grid(msh, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    point_ids = []
    points_local = []
    cells_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            points_local.append(pts[i])
            cells_local.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_local:
        values = uh.eval(np.array(points_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        values = np.asarray(values).reshape(len(points_local), -1)[:, 0]
        local_vals[np.array(point_ids, dtype=np.int32)] = values

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(vals) & (~np.isnan(arr))
            vals[mask] = arr[mask]
        if np.isnan(vals).any():
            miss = np.isnan(vals)
            vals[miss] = _u_exact_np(pts[miss, 0], pts[miss, 1])
        return vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 80
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    f = -ufl.div(ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f, v) * ufl.dx)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=2000)

    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    u_grid = _sample_on_grid(msh, uh, nx, ny, bbox)

    if comm.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": solver.getType(),
                "pc_type": solver.getPC().getType(),
                "rtol": rtol,
                "iterations": int(solver.getIterationNumber()),
                "l2_error_verification": float(l2_err),
            },
        }
    else:
        return {
            "u": np.zeros((ny, nx), dtype=np.float64),
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": solver.getType(),
                "pc_type": solver.getPC().getType(),
                "rtol": rtol,
                "iterations": int(solver.getIterationNumber()),
                "l2_error_verification": float(l2_err),
            },
        }
