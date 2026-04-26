import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_numpy(x):
    return np.sin(np.pi * x[0] * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.array(vals, dtype=np.float64).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    degree = 2
    mesh_resolution = 24
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0] * x[1])
    f_expr = -ufl.div(ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f_expr, v) * ufl.dx)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0] * X[1]))
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

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    uh = fem.Function(V)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Accuracy verification: L2 error against manufactured exact solution
    V_err = fem.functionspace(domain, ("Lagrange", degree + 2))
    u_ex = fem.Function(V_err)
    u_ex.interpolate(lambda X: np.sin(np.pi * X[0] * X[1]))
    uh_high = fem.Function(V_err)
    uh_high.interpolate(uh)

    err_form = fem.form((uh_high - u_ex) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    iterations = int(solver.getIterationNumber())
    elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": iterations,
        "l2_error": float(l2_error),
        "wall_time_sec": float(elapsed),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
