import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: convection_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: high
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: oscillatory_rhs
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: supg
time_method: none
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: convection_diffusion
"""


def _beta_constant(domain):
    return fem.Constant(domain, np.array([3.0, 3.0], dtype=ScalarType))


def _source_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])


def _build_problem(n: int, degree: int = 1):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps = 0.05
    beta_vec = np.array([3.0, 3.0], dtype=np.float64)
    beta = _beta_constant(domain)
    f_expr = _source_expr(domain)

    h = ufl.CellDiameter(domain)
    beta_mag = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    pe = beta_mag * h / (2.0 * eps)
    tau = h / (2.0 * beta_mag) * (ufl.cosh(pe) / ufl.sinh(pe) - 1.0 / pe)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    return domain, V, a, L, [bc], beta_vec, eps


def _solve_once(n: int, degree: int = 1, rtol: float = 1e-9):
    domain, V, a, L, bcs, beta_vec, eps = _build_problem(n, degree)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=4000)
    solver.setFromOptions()

    try:
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solver failed")
    except Exception:
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    beta = _beta_constant(domain)
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
    strong_res = ufl.dot(beta, ufl.grad(uh)) - eps * ufl.div(ufl.grad(uh)) - f_expr
    residual_sq = fem.assemble_scalar(fem.form(ufl.inner(strong_res, strong_res) * ufl.dx))
    residual_sq = domain.comm.allreduce(residual_sq, op=MPI.SUM)
    residual_l2 = math.sqrt(max(residual_sq, 0.0))

    return {
        "domain": domain,
        "u": uh,
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "residual_l2": float(residual_l2),
    }


def _probe_function(u_func, pts3):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts3)
    if u_func.function_space.mesh.comm.rank == 0:
        return np.nan_to_num(vals.reshape(ny, nx), nan=0.0)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]
    candidate_ns = [72, 96, 128, 160, 192]

    best = None
    prev_grid = None
    conv_indicator = None

    for i, n in enumerate(candidate_ns):
        result = _solve_once(n, degree=1, rtol=1e-9)
        u_grid = _sample_on_grid(result["u"], grid_spec)

        if comm.rank == 0:
            if prev_grid is not None:
                diff = np.linalg.norm(u_grid - prev_grid)
                ref = max(np.linalg.norm(u_grid), 1e-14)
                conv_indicator = float(diff / ref)
            prev_grid = u_grid.copy()

        conv_indicator = comm.bcast(conv_indicator, root=0)
        best = (result, u_grid)

        elapsed = time.perf_counter() - t0
        if i > 1 and conv_indicator is not None and conv_indicator < 1.8e-3 and elapsed > 20.0:
            break
        if elapsed > 50.0:
            break

    result, u_grid = best
    solver_info = {
        "mesh_resolution": result["mesh_resolution"],
        "element_degree": result["element_degree"],
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": result["rtol"],
        "iterations": result["iterations"],
        "accuracy_verification": {
            "residual_l2": result["residual_l2"],
            "grid_convergence_indicator": conv_indicator,
            "stabilization": "SUPG",
        },
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
