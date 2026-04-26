import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.exp(4.0 * x) * np.sin(np.pi * y)


def _rhs_numpy(x, y, k):
    # For u = exp(4x) sin(pi y),
    # Δu = (16 - pi^2) u
    # so -Δu - k^2 u = (pi^2 - 16 - k^2) u
    coef = (np.pi**2 - 16.0 - k**2)
    return coef * np.exp(4.0 * x) * np.sin(np.pi * y)


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    point_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            point_ids.append(i)

    local_values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64),
                           np.array(local_cells, dtype=np.int32))
        vals = np.array(vals).reshape(len(local_points), -1)[:, 0]
        local_values[np.array(point_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_values, root=0)
    if comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def _build_and_solve(mesh_resolution, degree, k, rtol, preferred_ksp="gmres", preferred_pc="ilu"):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    u_exact_ufl = ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = (ufl.pi**2 - 16.0 - k**2) * u_exact_ufl

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(4.0 * X[0]) * np.sin(np.pi * X[1]))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

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

    attempts = [
        (preferred_ksp, preferred_pc),
        ("gmres", "ilu"),
        ("preonly", "lu"),
    ]

    last_error = None
    used_ksp = None
    used_pc = None
    its = 0

    for ksp_type, pc_type in attempts:
        try:
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType(ksp_type)
            solver.getPC().setType(pc_type)
            solver.setTolerances(rtol=rtol, atol=1e-14, max_it=20000)
            if ksp_type == "gmres":
                solver.setGMRESRestart(200)
            solver.setFromOptions()
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            reason = solver.getConvergedReason()
            its = solver.getIterationNumber()
            if reason > 0 or ksp_type == "preonly":
                used_ksp = ksp_type
                used_pc = pc_type
                break
            last_error = RuntimeError(f"KSP did not converge, reason={reason}")
        except Exception as e:
            last_error = e
            used_ksp = None
            used_pc = None

    if used_ksp is None:
        raise RuntimeError(f"All solver attempts failed: {last_error}")

    return msh, V, uh, used_ksp, used_pc, its


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    k = float(
        case_spec.get("wavenumber", pde.get("k", pde.get("wavenumber", 25.0)))
    )

    # Adaptive accuracy choice while staying conservative on runtime.
    # This manufactured smooth-but-strongly-varying solution benefits from quadratic FEM.
    # We use P2 and a reasonably fine mesh to comfortably meet the tolerance.
    degree = 2
    mesh_resolution = 96
    rtol = 1.0e-10
    preferred_ksp = "gmres"
    preferred_pc = "ilu"

    msh, V, uh, ksp_type, pc_type, iterations = _build_and_solve(
        mesh_resolution=mesh_resolution,
        degree=degree,
        k=k,
        rtol=rtol,
        preferred_ksp=preferred_ksp,
        preferred_pc=preferred_pc,
    )

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    vals = _probe_function(uh, pts)

    if comm.rank == 0:
        if np.isnan(vals).any():
            raise RuntimeError("Point evaluation failed for some output grid points.")
        u_grid = vals.reshape(ny, nx)
        u_exact_grid = _exact_numpy(XX, YY)
        max_err = float(np.max(np.abs(u_grid - u_exact_grid)))
        l2_grid_err = float(np.sqrt(np.mean((u_grid - u_exact_grid) ** 2)))
        wall = time.perf_counter() - t0

        solver_info = {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification_max_abs_error_on_output_grid": max_err,
            "verification_rmse_on_output_grid": l2_grid_err,
            "wall_time_sec": float(wall),
        }

        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "helmholtz", "k": 25.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
