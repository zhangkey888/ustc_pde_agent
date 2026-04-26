import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _manufactured(domain):
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    kappa = 1.0 + 0.4 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact))
    return u_exact, kappa, f


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(ids), -1)[:, 0]
        values[np.asarray(ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(values, root=0)
    if msh.comm.rank == 0:
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    n = 64
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_exact, kappa, f = _manufactured(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_solver_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
    ksp.setFromOptions()
    tmp = uh.x.petsc_vec.duplicate()
    tmp.set(0.0)
    ksp.solve(b, tmp)
    iterations = int(ksp.getIterationNumber())

    l2_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(l2_form), op=MPI.SUM))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    u_grid = _probe_function(uh, pts).reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": n,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "l2_error_verification": float(l2_error),
        },
    }
