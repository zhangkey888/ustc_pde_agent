import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    material = pde.get("material", {})
    E = float(material.get("E", 1.0))
    nu = float(material.get("nu", 0.45))

    degree = 2 if nu > 0.4 else 1
    mesh_resolution = 48 if nu > 0.4 else 40

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
            ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    uh = fem.Function(V)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    rtol = 1.0e-10
    try:
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("cg")
        pc = ksp.getPC()
        pc.setType("hypre")
        ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("KSP did not converge")
        uh.x.scatter_forward()
        ksp_type = ksp.getType()
        pc_type = pc.getType()
        iterations = int(ksp.getIterationNumber())
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            u=uh,
            petsc_options_prefix="elasticity_fallback_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"
        iterations = 1

    err2 = fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx))
    norm2 = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err2, op=MPI.SUM))
    norm_L2 = np.sqrt(comm.allreduce(norm2, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = vals

    if comm.size > 1:
        local_vals = np.where(np.isnan(values), -1.0e300, values)
        global_vals = np.empty_like(local_vals)
        comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
        values = global_vals

    mag = np.linalg.norm(values, axis=1).reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(err_L2),
        "relative_l2_error": float(rel_L2),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": mag, "solver_info": solver_info}
