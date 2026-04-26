import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0], dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            raise RuntimeError("Some output grid points could not be evaluated on any MPI rank.")
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec["output"]
    grid_spec = output["grid"]

    kappa_value = float(case_spec.get("coefficients", {}).get("kappa", 1.0))
    target_time = 0.88

    # DIAGNOSIS
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

    # METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: hypre
    # special_treatment: none
    # pde_skill: poisson

    # Accuracy/time trade-off: start with a reasonably fine P2 mesh, increase if comfortably below budget.
    mesh_resolution = int(case_spec.get("mesh_resolution", 96))
    element_degree = int(case_spec.get("element_degree", 2))

    # If user/benchmark didn't provide parameters, push accuracy within expected time budget.
    if "mesh_resolution" not in case_spec:
        mesh_resolution = 160
    if "element_degree" not in case_spec:
        element_degree = 2

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = 2.0 * (ufl.pi ** 2) * ScalarType(kappa_value) * u_exact_ufl

    uD = fem.Function(V)
    uD.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(kappa_value))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    try:
        pc.setType("hypre")
    except Exception:
        pc.setType("jacobi")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=10000)
    solver.setFromOptions()

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    its = int(solver.getIterationNumber())
    ksp_type = solver.getType()
    pc_type = solver.getPC().getType()
    rtol = float(solver.getTolerances()[0])

    # Accuracy verification
    err_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)), op=MPI.SUM))
    u_exact_interp = fem.Function(V)
    u_exact_interp.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))
    local_max = 0.0
    if uh.x.array.size:
        local_max = np.max(np.abs(uh.x.array - u_exact_interp.x.array))
    err_max = comm.allreduce(local_max, op=MPI.MAX)

    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": rtol,
        "iterations": its,
        "verification_L2_error": float(err_L2),
        "verification_max_dof_error": float(err_max),
        "wall_time_sec": float(elapsed),
        "case_id": case_spec.get("case_id", pde.get("case_id", "poisson_p2")),
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "case_id": "poisson_p2",
        "pde": {"type": "poisson", "time": None},
        "coefficients": {"kappa": 1.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
