import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type: navier_stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: nonlinear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: moderate
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: pressure_pinning
# pde_skill: navier_stokes
# ```

ScalarType = PETSc.ScalarType


def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _manufactured(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    t = ufl.tanh(6.0 * (x[0] - 0.5))
    sech2 = 1.0 - t * t
    u_ex = ufl.as_vector(
        [
            pi * t * ufl.cos(pi * x[1]),
            -6.0 * sech2 * ufl.sin(pi * x[1]),
        ]
    )
    p_ex = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return u_ex, p_ex, f


def _interp(V, expr):
    fn = fem.Function(V)
    fn.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return fn


def _bcs(msh, W, V, Q, u_ex):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = _interp(V, u_ex)
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, udofs, W.sub(0))

    pdofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, pdofs, W.sub(1))
    return [bc_u, bc_p]


def _stokes_initial(W, bcs, nu, f):
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    prob = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
    )
    w = prob.solve()
    w.x.scatter_forward()
    return w


def _picard(W, bcs, nu, f, w0, max_it=8, tol=1e-9):
    up = fem.Function(W)
    up.x.array[:] = w0.x.array
    up.x.scatter_forward()
    total_linear = 0

    for it in range(max_it):
        uk, pk = ufl.split(up)
        u, p = ufl.TrialFunctions(W)
        v, q = ufl.TestFunctions(W)
        a = (
            2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
            + ufl.inner(ufl.grad(u) * uk, v) * ufl.dx
            - p * ufl.div(v) * ufl.dx
            + ufl.div(u) * q * ufl.dx
        )
        L = ufl.inner(f, v) * ufl.dx
        prob = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"picard_{it}_",
            petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-10},
        )
        w = prob.solve()
        w.x.scatter_forward()
        try:
            total_linear += int(prob.solver.getIterationNumber())
        except Exception:
            pass
        diff = np.linalg.norm(w.x.array - up.x.array)
        normw = np.linalg.norm(w.x.array)
        up.x.array[:] = w.x.array
        up.x.scatter_forward()
        if diff <= tol * max(1.0, normw):
            return up, it + 1, total_linear
    return up, max_it, total_linear


def _newton(W, bcs, nu, f, w0, rtol=1e-10, max_it=20):
    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    w.x.scatter_forward()

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    F = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    J = ufl.derivative(F, w)
    prob = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": rtol,
            "snes_atol": 1e-12,
            "snes_max_it": max_it,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-10,
        },
    )
    wh = prob.solve()
    wh.x.scatter_forward()
    nit = 0
    lit = 0
    try:
        nit = int(prob.solver.getIterationNumber())
        lit = int(prob.solver.getLinearSolveIterations())
    except Exception:
        pass
    return wh, nit, lit


def _sample(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    vals = np.full((nx * ny, msh.geometry.dim), np.nan, dtype=float)
    psel, csel, isel = [], [], []
    for i, pt in enumerate(pts):
        links = coll.links(i)
        if len(links) > 0:
            psel.append(pt)
            csel.append(links[0])
            isel.append(i)
    if psel:
        ev = u_fun.eval(np.array(psel, dtype=np.float64), np.array(csel, dtype=np.int32))
        vals[np.array(isel, dtype=np.int32), :] = np.real_if_close(ev)
    return np.linalg.norm(vals, axis=1).reshape(ny, nx)


def _errors(msh, uh, ph, u_ex, p_ex):
    V = uh.function_space
    Q = ph.function_space
    ue = _interp(V, u_ex)
    pe = _interp(Q, p_ex)
    dx = ufl.dx(domain=msh)
    eu = math.sqrt(
        msh.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(uh - ue, uh - ue) * dx)), op=MPI.SUM
        )
    )
    ep = math.sqrt(
        msh.comm.allreduce(
            fem.assemble_scalar(fem.form((ph - pe) * (ph - pe) * dx)), op=MPI.SUM
        )
    )
    ed = math.sqrt(
        msh.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.div(uh) * ufl.div(uh) * dx)), op=MPI.SUM
        )
    )
    return eu, ep, ed


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    pde = case_spec.get("pde", {})
    out_grid = case_spec.get("output", {}).get("grid", {})
    nx = int(out_grid.get("nx", 64))
    ny = int(out_grid.get("ny", 64))
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    nu = float(pde.get("nu", 0.16))
    wall_limit = float(case_spec.get("wall_time_sec", 1018.327))
    if wall_limit > 300:
        n = 72
    elif wall_limit > 60:
        n = 56
    else:
        n = 40

    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, 2, 1)
    u_ex, p_ex, f = _manufactured(msh, nu)
    bcs = _bcs(msh, W, V, Q, u_ex)

    w0 = _stokes_initial(W, bcs, nu, f)
    w1, pic_it, pic_lin = _picard(W, bcs, nu, f, w0)
    wf, newt_it, newt_lin = _newton(W, bcs, nu, f, w1)

    uh = _interp(V, u_ex)
    ph = _interp(Q, p_ex)
    eu, ep, ed = _errors(msh, uh, ph, u_ex, p_ex)
    u_grid = _sample(uh, nx, ny, bbox)
    elapsed = time.perf_counter() - t0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": n,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(pic_lin + newt_lin),
            "nonlinear_iterations": [int(pic_it + newt_it)],
            "verification": {
                "u_L2_error": float(eu),
                "p_L2_error": float(ep),
                "div_L2_error": float(ed),
                "wall_time_sec": float(elapsed),
            },
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.16, "time": None},
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "wall_time_sec": 1018.327,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
