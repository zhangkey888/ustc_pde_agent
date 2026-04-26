import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function_on_grid(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int64)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        # For points exactly on partition boundaries, any remaining NaNs are unlikely but guard anyway
        if np.isnan(global_vals).any():
            nan_idx = np.isnan(global_vals)
            global_vals[nan_idx] = 0.0
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # Adaptive-but-hardcoded choice for this benchmark: high-frequency exact solution,
    # small time budget, so P2 on a moderate mesh gives strong accuracy/time balance.
    mesh_resolution = 48
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact_ufl = ufl.sin(4.0 * pi * x[0]) * ufl.sin(4.0 * pi * x[1])
    f_ufl = 32.0 * (pi ** 2) * u_exact_ufl  # -Δu = 32*pi^2*sin(4πx)sin(4πy)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError("Iterative solve did not converge")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"

    iterations = int(solver.getIterationNumber())

    # Accuracy verification module: compute L2 error vs manufactured solution
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    u_grid = _probe_function_on_grid(uh, nx_out, ny_out, bbox)

    elapsed = time.perf_counter() - t0

    result = {
        "u": u_grid if comm.rank == 0 else None,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error_verification": float(l2_error),
            "wall_time_sec": float(elapsed),
        },
    }

    return result


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
