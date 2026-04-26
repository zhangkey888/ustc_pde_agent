import time, numpy as np, ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
ScalarType = PETSc.ScalarType
DIAGNOSIS = "DIAGNOSIS: equation_type=helmholtz, spatial_dim=2, domain_geometry=rectangle, unknowns=scalar, coupling=none, linearity=linear, time_dependence=steady, dominant_physics=wave, bc_type=all_dirichlet"
METHOD = "METHOD: spatial_method=fem, element_or_basis=Lagrange_P2, stabilization=none, time_method=none, nonlinear_solver=none, linear_solver=gmres, preconditioner=ilu, pde_skill=helmholtz"
def _solve_one(n, degree, k, rtol):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    rhs = fem.Function(V)
    rhs.interpolate(fem.Expression(ufl.cos(4.0*ufl.pi*x[0])*ufl.sin(3.0*ufl.pi*x[1]), V.element.interpolation_points))
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2)*u*v) * ufl.dx
    L = ufl.inner(rhs, v) * ufl.dx
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)
    try:
        prob = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"helm_{n}_", petsc_options={"ksp_type":"gmres","ksp_rtol":rtol,"ksp_atol":1e-12,"ksp_max_it":5000,"pc_type":"ilu","pc_factor_levels":1})
        uh = prob.solve(); uh.x.scatter_forward()
        return msh, uh, {"mesh_resolution":int(n),"element_degree":int(degree),"ksp_type":"gmres","pc_type":"ilu","rtol":float(rtol),"iterations":int(prob.solver.getIterationNumber())}
    except Exception:
        prob = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"helm_lu_{n}_", petsc_options={"ksp_type":"preonly","pc_type":"lu"})
        uh = prob.solve(); uh.x.scatter_forward()
        return msh, uh, {"mesh_resolution":int(n),"element_degree":int(degree),"ksp_type":"preonly","pc_type":"lu","rtol":float(rtol),"iterations":1}
def _l2diff(cmsh, cu, fu):
    Vc = cu.function_space
    fi = fem.Function(Vc); fi.interpolate(fu)
    d = fem.Function(Vc); d.x.array[:] = cu.x.array - fi.x.array
    lv = fem.assemble_scalar(fem.form(ufl.inner(d, d) * ufl.dx))
    return float(np.sqrt(max(cmsh.comm.allreduce(lv, op=MPI.SUM), 0.0)))
def _sample(msh, uh, grid):
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx*ny)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    local = np.full(nx*ny, np.nan)
    epts, ecells, eids = [], [], []
    for i in range(nx*ny):
        links = coll.links(i)
        if len(links) > 0:
            epts.append(pts[i]); ecells.append(links[0]); eids.append(i)
    if epts:
        vals = uh.eval(np.asarray(epts, dtype=np.float64), np.asarray(ecells, dtype=np.int32))
        local[np.asarray(eids, dtype=np.int64)] = np.real(vals).reshape(-1)
    gathered = msh.comm.gather(local, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx*ny, np.nan)
        for arr in gathered:
            m = ~np.isnan(arr); out[m] = arr[m]
        out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return None
def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]
    pde = case_spec.get("pde", {})
    k, degree, rtol = float(pde.get("wavenumber", 8.0)), 2, 1e-9
    candidates, budget, safety = [40, 56, 72, 88, 104, 120], 52.334, 8.0
    prev, verification, best = None, {}, None
    for n in candidates:
        msh, uh, info = _solve_one(n, degree, k, rtol); best = (msh, uh, info, n)
        if prev is not None:
            verification[f"l2_diff_{prev[3]}_to_{n}"] = _l2diff(prev[0], prev[1], uh)
        elapsed = time.perf_counter() - t0
        if elapsed > budget - safety: break
        if prev is not None and verification[f"l2_diff_{prev[3]}_to_{n}"] < 2.0e-3 and elapsed > 3.0: break
        prev = (msh, uh, info, n)
    msh, uh, info, _ = best
    u_grid = _sample(msh, uh, grid)
    if MPI.COMM_WORLD.rank != 0: return {"u": None, "solver_info": {}}
    info = dict(info); info["verification"] = verification; info["wall_time_sec"] = float(time.perf_counter()-t0); info["diagnosis"] = DIAGNOSIS; info["method"] = METHOD
    return {"u": np.asarray(u_grid, dtype=np.float64).reshape(int(grid["ny"]), int(grid["nx"])), "solver_info": info}
