import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _source_expr(x):
    return np.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2)) + np.exp(
        -250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.70) ** 2)
    )


def _sample_on_grid(u_func, domain, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        # Points on the boundary/corners should still be evaluable, but clamp any
        # residual NaNs to zero since Dirichlet data are zero and bbox matches domain.
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape(ny, nx)
    return None


def _solve_poisson_once(mesh_resolution=96, degree=2, ksp_type="preonly", pc_type="lu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f.interpolate(_source_expr)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()
    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Primary linear solve did not converge")
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()
    its = int(solver.getIterationNumber())
    actual_ksp = solver.getType()
    actual_pc = solver.getPC().getType()
    return domain, uh, its, actual_ksp, actual_pc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]

    # Accuracy/time balance: use a reasonably fine quadratic FEM discretization.
    mesh_resolution = 96
    element_degree = 2
    requested_ksp = "preonly"
    requested_pc = "lu"
    rtol = 1e-10

    domain, uh, iterations, ksp_type, pc_type = _solve_poisson_once(
        mesh_resolution=mesh_resolution,
        degree=element_degree,
        ksp_type=requested_ksp,
        pc_type=requested_pc,
        rtol=rtol,
    )

    u_grid = _sample_on_grid(uh, domain, grid)

    # Accuracy verification: self-convergence estimate via coarser-vs-fine sampled-grid difference.
    # This is lightweight and does not require an analytical solution.
    verification = {}
    try:
        coarse_res = max(32, mesh_resolution // 2)
        domain_c, uh_c, _, _, _ = _solve_poisson_once(
            mesh_resolution=coarse_res,
            degree=element_degree,
            ksp_type=requested_ksp,
            pc_type=requested_pc,
            rtol=rtol,
        )
        u_grid_c = _sample_on_grid(uh_c, domain_c, grid)
        if comm.rank == 0:
            diff = u_grid - u_grid_c
            verification = {
                "verification_type": "grid_self_convergence",
                "coarse_mesh_resolution": coarse_res,
                "linf_diff_sampled": float(np.max(np.abs(diff))),
                "l2_diff_sampled": float(np.sqrt(np.mean(diff**2))),
            }
    except Exception as exc:
        if comm.rank == 0:
            verification = {"verification_type": "failed", "message": str(exc)}

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
        }
        solver_info.update(verification)
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": {}}


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
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
