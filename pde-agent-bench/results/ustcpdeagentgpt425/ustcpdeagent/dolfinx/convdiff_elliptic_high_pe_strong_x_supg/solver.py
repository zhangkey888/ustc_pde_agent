import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "equation_type=convection_diffusion; spatial_dim=2; domain_geometry=rectangle; unknowns=scalar; coupling=none; linearity=linear; time_dependence=steady; dominant_physics=mixed; peclet_or_reynolds=high; solution_regularity=smooth; bc_type=all_dirichlet; special_notes=manufactured_solution"
METHOD = "spatial_method=fem; element_or_basis=Lagrange_P2; stabilization=supg; time_method=none; nonlinear_solver=none; linear_solver=gmres; preconditioner=ilu; special_treatment=none; pde_skill=convection_diffusion"


def _probe_function(u_func: fem.Function, pts: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)
    return values


def _sample_to_grid(u_func: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _exact_grid(grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)


def _assemble_and_solve(n: int, degree: int = 2):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    eps_value = 0.01
    beta = fem.Constant(msh, np.array([15.0, 0.0], dtype=np.float64))

    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = -eps_value * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    tau = h / (2.0 * beta_norm)

    a = (
        eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("iterative solve failed")
    except Exception:
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    e = uh - u_bc
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_sq = comm.allreduce(l2_sq, op=MPI.SUM)
    l2_error = float(np.sqrt(l2_sq))
    return uh, l2_error, ksp


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    start = time.perf_counter()
    candidates = [64, 96, 128]
    chosen = None
    for n in candidates:
        uh, l2_error, ksp = _assemble_and_solve(n, degree=2)
        elapsed = time.perf_counter() - start
        chosen = (n, uh, l2_error, ksp, elapsed)
        if elapsed > 2.1:
            break
    n, uh, l2_error, ksp, elapsed = chosen
    u_grid = _sample_to_grid(uh, grid)
    exact_grid = _exact_grid(grid)
    sample_linf_error = float(np.max(np.abs(u_grid - exact_grid)))
    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": l2_error,
        "sample_linf_error": sample_linf_error,
        "diagnosis": DIAGNOSIS,
        "method": METHOD,
        "wall_time_estimate": float(elapsed),
    }
    return {"u": u_grid, "solver_info": solver_info}
