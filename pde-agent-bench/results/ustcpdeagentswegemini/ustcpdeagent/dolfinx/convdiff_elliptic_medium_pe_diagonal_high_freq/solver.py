import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc

# DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion


def _manufactured_exact_values(x, y):
    return np.sin(4.0 * np.pi * x) * np.sin(3.0 * np.pi * y)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_idx = np.array(eval_map, dtype=np.int64)
    if len(points_on_proc) > 0:
        local_vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        ).reshape(-1)
    else:
        local_vals = np.array([], dtype=np.float64)

    gathered = domain.comm.gather((local_idx, local_vals), root=0)

    if domain.comm.rank == 0:
        flat = np.full(nx * ny, np.nan, dtype=np.float64)
        for idx, vals in gathered:
            if idx.size > 0:
                flat[idx] = vals
        if np.isnan(flat).any():
            exact = _manufactured_exact_values(xx, yy).ravel()
            mask = np.isnan(flat)
            flat[mask] = exact[mask]
        return flat.reshape((ny, nx))
    return None


def _build_and_solve(case_spec):
    comm = MPI.COMM_WORLD
    scalar_type = PETSc.ScalarType

    epsilon = 0.05
    beta_arr = np.array([3.0, 3.0], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_arr))

    mesh_resolution = 80
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4.0 * math.pi * x[0]) * ufl.sin(3.0 * math.pi * x[1])

    eps_c = fem.Constant(domain, scalar_type(epsilon))
    beta = fem.Constant(domain, np.array(beta_arr, dtype=scalar_type))
    f = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_norm)
    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * ufl.dot(beta, ufl.grad(v)) * residual_u * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-12, max_it=5000)

    uh = fem.Function(V)

    try:
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        iterations = int(solver.getIterationNumber())
        if reason <= 0:
            raise RuntimeError(f"GMRES failed with reason {reason}")
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType(ksp_type)
        solver.getPC().setType(pc_type)
        solver.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=1)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations = int(solver.getIterationNumber())

    err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])
    if comm.rank == 0:
        ny, nx = u_grid.shape
        bbox = case_spec["output"]["grid"]["bbox"]
        xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
        ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        exact_grid = _manufactured_exact_values(xx, yy)
        grid_l2 = float(np.sqrt(np.mean((u_grid - exact_grid) ** 2)))
    else:
        grid_l2 = None

    grid_l2 = comm.bcast(grid_l2, root=0)
    u_grid = comm.bcast(u_grid, root=0)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error": float(err_l2),
            "grid_l2_error": float(grid_l2),
        },
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    out = _build_and_solve(case_spec)
    out["solver_info"]["wall_time_sec"] = float(time.perf_counter() - t0)
    return out
