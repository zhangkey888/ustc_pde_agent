import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
ScalarType = PETSc.ScalarType
DIAGNOSIS_AND_METHOD = "DIAGNOSIS: heat transient manufactured; METHOD: FEM P1 backward_euler CG+hypre"
def _times(case_spec):
    ts = case_spec.get("pde", {}).get("time", {})
    return float(ts.get("t0", 0.0)), float(ts.get("t_end", 0.2)), float(ts.get("dt", 0.02))
def _uex(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-0.5*t) * ufl.sin(2.0*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
def _src(msh, kappa, t):
    u = _uex(msh, t)
    return (-0.5 + 5.0*kappa*ufl.pi*ufl.pi) * u
def _interp(V, expr):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f
def _sample(msh, uh, grid):
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    vals = np.full(nx*ny, np.nan)
    p, c, m = [], [], []
    for i in range(nx*ny):
        links = coll.links(i)
        if len(links) > 0:
            p.append(pts[i]); c.append(links[0]); m.append(i)
    if p:
        out = uh.eval(np.array(p, dtype=np.float64), np.array(c, dtype=np.int32)).reshape(-1)
        vals[np.array(m, dtype=np.int32)] = out
    allv = msh.comm.allgather(vals)
    g = np.full_like(vals, np.nan)
    for a in allv:
        mask = np.isnan(g) & ~np.isnan(a)
        g[mask] = a[mask]
    return np.nan_to_num(g, nan=0.0).reshape(ny, nx)
def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0, t_end, dt = _times(case_spec)
    kappa_val = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 0.1))
    mesh_resolution, degree = 24, 1
    n_steps = max(1, int(round((t_end - t0)/dt)))
    dt = (t_end - t0)/n_steps
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_n = _interp(V, _uex(msh, ScalarType(t0)))
    uD = _interp(V, _uex(msh, ScalarType(t0+dt)))
    bc = fem.dirichletbc(uD, dofs)
    f_fun = _interp(V, _src(msh, kappa_val, ScalarType(t0+dt)))
    kappa = fem.Constant(msh, ScalarType(kappa_val))
    dtc = fem.Constant(msh, ScalarType(dt))
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = (u*v + dtc*kappa*ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n*v + dtc*f_fun*v) * ufl.dx
    a_form, L_form = fem.form(a), fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc]); A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A); solver.setType("cg"); solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    uh = fem.Function(V); iterations = 0
    for step in range(1, n_steps+1):
        t = t0 + step*dt
        uD.interpolate(fem.Expression(_uex(msh, ScalarType(t)), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_src(msh, kappa_val, ScalarType(t)), V.element.interpolation_points))
        with b.localForm() as loc: loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array; u_n.x.scatter_forward()
    uT = _interp(V, _uex(msh, ScalarType(t_end)))
    err = fem.Function(V); err.x.array[:] = uh.x.array - uT.x.array; err.x.scatter_forward()
    l2sq = comm.allreduce(fem.assemble_scalar(fem.form(err*err*ufl.dx)), op=MPI.SUM)
    l2err = math.sqrt(max(l2sq, 0.0))
    grid = case_spec["output"]["grid"]
    return {"u": _sample(msh, uh, grid), "u_initial": _sample(msh, _interp(V, _uex(msh, ScalarType(t0))), grid), "solver_info": {"mesh_resolution": mesh_resolution, "element_degree": degree, "ksp_type": solver.getType(), "pc_type": solver.getPC().getType(), "rtol": 1e-8, "iterations": iterations, "dt": dt, "n_steps": n_steps, "time_scheme": "backward_euler", "accuracy_verification": {"manufactured_solution": True, "l2_error": l2err}}}
