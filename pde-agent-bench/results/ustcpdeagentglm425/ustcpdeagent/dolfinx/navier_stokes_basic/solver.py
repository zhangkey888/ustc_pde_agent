import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    nu = pde["nu"]
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 30
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u1_ex = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
    u2_ex = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    u_ex = ufl.as_vector([u1_ex, u2_ex])
    
    grad_u_ex = ufl.grad(u_ex)
    f_source = grad_u_ex * u_ex - nu * ufl.div(grad_u_ex)
    
    # BCs
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(ufl.as_vector([u1_ex, u2_ex]), V.element.interpolation_points))
    
    bf = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, bf)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q); p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Stokes solve
    (ut, pt) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a_s = (2*nu*ufl.inner(ufl.sym(ufl.grad(ut)), ufl.sym(ufl.grad(v))) - pt*ufl.div(v) + ufl.div(ut)*q)*ufl.dx
    L_s = ufl.inner(f_source, v)*ufl.dx
    
    w_s = fem.Function(W)
    sp = petsc.LinearProblem(a_s, L_s, bcs=bcs, u=w_s,
        petsc_options={"ksp_type": "minres", "pc_type": "hypre", "ksp_rtol": "1e-10"},
        petsc_options_prefix="stokes_")
    w_s = sp.solve()
    w_s.x.scatter_forward()
    
    # Newton solve
    w = fem.Function(W)
    w.x.array[:] = w_s.x.array[:]
    w.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    def eps(uv): return ufl.sym(ufl.grad(uv))
    
    F = (2*nu*ufl.inner(eps(u),eps(v)) - p*ufl.div(v) + ufl.div(u)*q + ufl.inner(ufl.grad(u)*u,v) - ufl.inner(f_source,v))*ufl.dx
    J = ufl.derivative(F, w)
    
    ns = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_",
        petsc_options={"snes_type":"newtonls","snes_linesearch_type":"bt","snes_rtol":1e-8,"snes_atol":1e-10,"snes_max_it":30,"ksp_type":"gmres","pc_type":"hypre","ksp_rtol":"1e-10"})
    
    w_h = ns.solve()
    w_h.x.scatter_forward()
    
    u_h = w_h.sub(0).collapse()
    
    # Error
    ue = fem.Function(V)
    ue.interpolate(fem.Expression(ufl.as_vector([u1_ex, u2_ex]), V.element.interpolation_points))
    esq = fem.assemble_scalar(fem.form(ufl.inner(u_h-ue, u_h-ue)*ufl.dx))
    esq = comm.allreduce(esq, op=MPI.SUM)
    
    try: nit = int(ns._snes.getIterationNumber())
    except: nit = 0
    try: sit = int(sp.solver.getIterationNumber()) + int(ns._snes.getLinearSolveIterations())
    except: sit = 0
    
    # Sample grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])
    
    bb = geometry.bb_tree(domain, tdim)
    cc = geometry.compute_collisions_points(bb, pts.T)
    ic = geometry.compute_colliding_cells(domain, cc, pts.T)
    
    pp, cp, em = [], [], []
    for i in range(pts.shape[1]):
        lk = ic.links(i)
        if len(lk) > 0:
            pp.append(pts.T[i]); cp.append(lk[0]); em.append(i)
    
    uv = np.full((pts.shape[1], gdim), np.nan)
    if pp:
        vals = u_h.eval(np.array(pp), np.array(cp, dtype=np.int32))
        for idx, pos in enumerate(em): uv[pos] = vals[idx]
    
    uv_local = np.where(np.isnan(uv), 0.0, uv)
    uv_global = np.zeros_like(uv)
    comm.Allreduce(uv_local, uv_global, op=MPI.SUM)
    
    u_grid = np.linalg.norm(uv_global, axis=1).reshape(ny_out, nx_out)
    
    return {"u": u_grid, "solver_info": {
        "mesh_resolution": mesh_res, "element_degree": 2,
        "ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8,
        "iterations": sit, "nonlinear_iterations": [nit]
    }}
