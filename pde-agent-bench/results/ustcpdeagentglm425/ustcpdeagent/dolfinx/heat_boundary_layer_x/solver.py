import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    coeffs = pde.get("coefficients", {})
    kappa = float(coeffs.get("kappa", 1.0))
    ti = pde["time"]
    t0, t_end, dt_sug = float(ti["t0"]), float(ti["t_end"]), float(ti["dt"])
    
    gr = case_spec["output"]["grid"]
    nx_o, ny_o = gr["nx"], gr["ny"]
    bbox = gr["bbox"]
    
    # Use half suggested dt for better accuracy (O(dt) scheme)
    N = 32
    deg = 2
    dt = dt_sug / 2.0
    ns = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / ns
    
    dom = mesh.create_unit_square(comm, N, N)
    V = fem.functionspace(dom, ("Lagrange", deg))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    un = fem.Function(V)
    us = fem.Function(V)
    ff = fem.Function(V)
    gf = fem.Function(V)
    
    un.interpolate(lambda x: np.exp(5.*x[0]) * np.sin(np.pi*x[1]))
    
    a = ufl.inner(u,v)*ufl.dx + dt*kappa*ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
    L = ufl.inner(un,v)*ufl.dx + dt*ufl.inner(ff,v)*ufl.dx
    
    fd = dom.topology.dim - 1
    bf = mesh.locate_entities_boundary(dom, fd, lambda x: np.ones(x.shape[1],dtype=bool))
    bd = fem.locate_dofs_topological(V, fd, bf)
    bc = fem.dirichletbc(gf, bd)
    
    af = fem.form(a)
    Lf = fem.form(L)
    
    A = petsc.assemble_matrix(af, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(Lf.function_spaces)
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol, atol=1e-12)
    ksp.setFromOptions()
    
    tot_it = 0
    t = t0
    for _ in range(ns):
        tn = t + dt
        _tn = tn
        ff.interpolate(lambda x, _t=_tn: (np.pi**2-26.)*np.exp(-_t)*np.exp(5.*x[0])*np.sin(np.pi*x[1]))
        gf.interpolate(lambda x, _t=_tn: np.exp(-_t)*np.exp(5.*x[0])*np.sin(np.pi*x[1]))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, Lf)
        petsc.apply_lifting(b, [af], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, us.x.petsc_vec)
        us.x.scatter_forward()
        tot_it += ksp.getIterationNumber()
        un.x.array[:] = us.x.array[:]
        t = tn
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_o)
    ys = np.linspace(bbox[2], bbox[3], ny_o)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_o*ny_o)])
    
    tree = geometry.bb_tree(dom, dom.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    cells = geometry.compute_colliding_cells(dom, cand, pts)
    
    pp, cc, mm = [], [], []
    for i in range(pts.shape[0]):
        lk = cells.links(i)
        if len(lk) > 0:
            pp.append(pts[i]); cc.append(lk[0]); mm.append(i)
    
    u_grid = np.full((ny_o, nx_o), np.nan)
    if pp:
        pa = np.array(pp)
        ca = np.array(cc, dtype=np.int32)
        v = us.eval(pa, ca).flatten()
        fv = np.full(pts.shape[0], np.nan)
        fv[mm] = v
        u_grid = fv.reshape(ny_o, nx_o)
    
    ui_grid = np.full((ny_o, nx_o), np.nan)
    if pp:
        v0 = np.exp(5.*pa[:,0]) * np.sin(np.pi*pa[:,1])
        f0 = np.full(pts.shape[0], np.nan)
        f0[mm] = v0
        ui_grid = f0.reshape(ny_o, nx_o)
    
    # L2 error
    ue = fem.Function(V)
    ue.interpolate(lambda x: np.exp(-t_end)*np.exp(5.*x[0])*np.sin(np.pi*x[1]))
    err_f = fem.form(ufl.inner(us-ue,us-ue)*ufl.dx)
    l2e = np.sqrt(comm.allreduce(fem.assemble_scalar(err_f), op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2err={l2e:.4e} N={N} deg={deg} dt={dt}")
    
    return {"u": u_grid, "u_initial": ui_grid,
            "solver_info": {"mesh_resolution": N, "element_degree": deg,
                           "ksp_type": "cg", "pc_type": "jacobi", "rtol": rtol,
                           "iterations": tot_it, "dt": dt, "n_steps": ns,
                           "time_scheme": "backward_euler"}}

if __name__ == "__main__":
    r = solve({"pde":{"coefficients":{"kappa":1.0},"time":{"t0":0.0,"t_end":0.08,"dt":0.008,"scheme":"backward_euler"}},
               "output":{"grid":{"nx":50,"ny":50,"bbox":[0.,1.,0.,1.]}}})
    print(f"shape={r['u'].shape} max={np.nanmax(r['u']):.4e}")
