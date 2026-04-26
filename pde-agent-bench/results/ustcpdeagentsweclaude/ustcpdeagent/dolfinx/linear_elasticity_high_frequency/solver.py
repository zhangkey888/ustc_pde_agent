import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

def _lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam

def _eps(u):
    return ufl.sym(ufl.grad(u))

def _sigma(u, mu, lam, gdim):
    return 2.0 * mu * _eps(u) + lam * ufl.tr(_eps(u)) * ufl.Identity(gdim)

def _choose_resolution(case_spec, degree):
    grid = case_spec.get("output", {}).get("grid", {})
    base = max(int(grid.get("nx", 64)), int(grid.get("ny", 64)))
    return max(80, min(128, int(1.5 * base) if degree >= 2 else 2 * base))

def _manufactured(msh, E, nu):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        ufl.sin(4.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1]),
        ufl.cos(3.0 * pi * x[0]) * ufl.sin(4.0 * pi * x[1]),
    ])
    mu, lam = _lame_parameters(E, nu)
    f = -ufl.div(_sigma(u_exact, mu, lam, msh.geometry.dim))
    return u_exact, f, mu, lam

def _sample_values(u_fun, pts):
    msh = u_fun.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)
    vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    p_local = []
    c_local = []
    idx = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(pts[i])
            c_local.append(links[0])
            idx.append(i)
    if p_local:
        ev = u_fun.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        vals[np.array(idx, dtype=np.int32)] = np.asarray(ev, dtype=np.float64)
    return vals

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    E = float(pde.get("E", 1.0))
    nu = float(pde.get("nu", 0.28))

    degree = 2
    n = _choose_resolution(case_spec, degree)
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_exact, f, mu, lam = _manufactured(msh, E, nu)

    a = ufl.inner(_sigma(u, mu, lam, gdim), _eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    t0 = time.perf_counter()
    ksp_type_used, pc_type_used, rtol_used = "cg", "hypre", 1e-10
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix="elas_",
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "hypre",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-12,
                "ksp_max_it": 5000,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type_used = ksp.getType()
        pc_type_used = ksp.getPC().getType()
        rtol_used = float(ksp.getTolerances()[0])
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix="elas_lu_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type_used = ksp.getType()
        pc_type_used = ksp.getPC().getType()
        rtol_used = 0.0
    elapsed = time.perf_counter() - t0

    Vex = fem.functionspace(msh, ("Lagrange", degree + 2, (gdim,)))
    uex = fem.Function(Vex)
    uex.interpolate(fem.Expression(u_exact, Vex.element.interpolation_points))
    err = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    ref = fem.form(ufl.inner(uex, uex) * ufl.dx)
    l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(err), op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(fem.assemble_scalar(ref), op=MPI.SUM))
    rel_l2 = l2_err / l2_ref if l2_ref > 0 else l2_err

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    local = _sample_values(uh, pts)
    gathered = comm.gather(local, root=0)
    if rank == 0:
        vals = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(vals[:, 0]) & (~np.isnan(arr[:, 0]))
            vals[mask] = arr[mask]
        mags = np.linalg.norm(vals, axis=1).reshape(ny, nx)
    else:
        mags = None
    mags = comm.bcast(mags, root=0)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type_used),
        "pc_type": str(pc_type_used),
        "rtol": float(rtol_used),
        "iterations": int(iterations),
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2),
        "wall_time_sec": float(elapsed),
    }
    return {"u": mags, "solver_info": solver_info}
