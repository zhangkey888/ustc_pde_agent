import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0], dtype=np.float64)])

    bb = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb, pts3)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts3)

    local_pts = []
    local_cells = []
    local_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_pts.append(pts3[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        vals = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_pts), -1)[:, 0]
        local_vals[np.array(local_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at all requested grid points.")
        out = out.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    # Problem setup
    kappa = float(case_spec.get("coefficients", {}).get("kappa", 1.0))
    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    # Accuracy/time tuned defaults for this manufactured Poisson case
    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 28))
    element_degree = int(case_spec.get("agent_params", {}).get("element_degree", 2))
    rtol = float(case_spec.get("agent_params", {}).get("rtol", 1e-12))
    ksp_type = str(case_spec.get("agent_params", {}).get("ksp_type", "preonly"))
    pc_type = str(case_spec.get("agent_params", {}).get("pc_type", "lu"))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact_ufl = ufl.exp(2.0 * x[0]) * ufl.cos(pi * x[1])
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    def boundary_marker(X):
        return (
            np.isclose(X[0], 0.0)
            | np.isclose(X[0], 1.0)
            | np.isclose(X[1], 0.0)
            | np.isclose(X[1], 1.0)
        )

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(2.0 * X[0]) * np.cos(pi * X[1]))
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary_marker)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Solver info from underlying KSP if available
    iterations = None
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type_actual = ksp.getType()
        pc_type_actual = ksp.getPC().getType()
    except Exception:
        iterations = 0
        ksp_type_actual = ksp_type
        pc_type_actual = pc_type

    # Accuracy verification: L2 error against exact manufactured solution
    Vh = fem.functionspace(msh, ("Lagrange", max(element_degree + 2, 4)))
    u_exact_h = fem.Function(Vh)
    u_exact_h.interpolate(lambda X: np.exp(2.0 * X[0]) * np.cos(pi * X[1]))
    uh_h = fem.Function(Vh)
    uh_h.interpolate(uh)

    err_form = fem.form((uh_h - u_exact_h) ** 2 * ufl.dx)
    ref_form = fem.form((u_exact_h) ** 2 * ufl.dx)
    err_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    rel_L2 = err_L2 / ref_L2 if ref_L2 > 0 else err_L2

    u_grid = _sample_on_grid(uh, nx_out, ny_out, bbox)

    wall_time = time.perf_counter() - t0

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type_actual),
            "pc_type": str(pc_type_actual),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error": float(err_L2),
            "relative_l2_error": float(rel_L2),
            "wall_time_sec": float(wall_time),
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "coefficients": {"kappa": 1.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
