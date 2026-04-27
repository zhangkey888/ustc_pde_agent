import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    comm = MPI.COMM_WORLD
    
    nu_val = case_spec["pde"]["viscosity"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    N = 64
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()
    
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi_u = ufl.pi
    u_exact = ufl.as_vector([
        pi_u*ufl.cos(pi_u*x[1])*ufl.sin(pi_u*x[0]),
        -pi_u*ufl.cos(pi_u*x[0])*ufl.sin(pi_u*x[1])
    ])
    
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f_body = ufl.grad(u_exact)*u_exact - nu_c*ufl.div(ufl.grad(u_exact))
    
    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0))
    p0f = fem.Function(Q); p0f.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0f, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]
    
    # Stokes initial guess
    (us, ps) = ufl.TrialFunctions(W)
    (vs, qs) = ufl.TestFunctions(W)
    a_st = nu_c*ufl.inner(ufl.grad(us), ufl.grad(vs))*ufl.dx - ps*ufl.div(vs)*ufl.dx + ufl.div(us)*qs*ufl.dx
    L_st = ufl.inner(f_body, vs)*ufl.dx
    w_init = petsc.LinearProblem(a_st, L_st, bcs=bcs, petsc_options={"ksp_type":"preonly","pc_type":"lu"}, petsc_options_prefix="st_").solve()
    
    # Newton
    w = fem.Function(W)
    w.x.array[:] = w_init.x.array[:]; w.x.scatter_forward()
    (u_nl, p_nl) = ufl.split(w)
    (v_nl, q_nl) = ufl.TestFunctions(W)
    
    F_nl = (nu_c*ufl.inner(ufl.grad(u_nl), ufl.grad(v_nl))*ufl.dx
            + ufl.inner(ufl.grad(u_nl)*u_nl, v_nl)*ufl.dx
            - p_nl*ufl.div(v_nl)*ufl.dx + ufl.div(u_nl)*q_nl*ufl.dx
            - ufl.inner(f_body, v_nl)*ufl.dx)
    J_nl = ufl.derivative(F_nl, w)
    Fc = fem.form(F_nl); Jc = fem.form(J_nl)
    
    dw = fem.Function(W)
    nc = 0
    for nit in range(15):
        bv = petsc.assemble_vector(Fc)
        petsc.apply_lifting(bv, [Jc], bcs=[bcs], x0=[w.x.petsc_vec])
        bv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(bv, bcs, w.x.petsc_vec)
        rn = bv.norm()
        if rn < 1e-10:
            bv.destroy(); break
        Am = petsc.assemble_matrix(Jc, bcs=bcs); Am.assemble()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(Am); ksp.setType(PETSc.KSP.Type.PREONLY); ksp.getPC().setType(PETSc.PC.Type.LU)
        bv.scale(-1.0); ksp.solve(bv, dw.x.petsc_vec); dw.x.scatter_forward()
        w.x.array[:] += dw.x.array[:]; w.x.scatter_forward()
        nc += 1; ksp.destroy(); Am.destroy(); bv.destroy()
    
    u_h = w.sub(0).collapse()
    
    # Sample
    xmin,xmax,ymin,ymax = bbox
    xs = np.linspace(xmin,xmax,nx_out); ys = np.linspace(ymin,ymax,ny_out)
    XX,YY = np.meshgrid(xs,ys)
    pts = np.zeros((XX.size,3)); pts[:,0]=XX.ravel(); pts[:,1]=YY.ravel()
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    colls = geometry.compute_colliding_cells(msh, cands, pts)
    pp=[]; cp=[]; em=[]
    for i in range(len(pts)):
        lnk = colls.links(i)
        if len(lnk)>0: pp.append(pts[i]); cp.append(lnk[0]); em.append(i)
    
    uv = np.full((len(pts),gdim), np.nan)
    if pp:
        ev = u_h.eval(np.array(pp), np.array(cp, dtype=np.int32))
        for idx,gi in enumerate(em): uv[gi] = ev[idx]
    
    vmag = np.sqrt(uv[:,0]**2+uv[:,1]**2)
    u_grid = vmag.reshape(ny_out, nx_out)
    
    # Error verification
    u_ex_f = fem.Function(V)
    u_ex_f.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    ef = fem.form(ufl.inner(u_h-u_ex_f, u_h-u_ex_f)*ufl.dx)
    eg = np.sqrt(comm.allreduce(fem.assemble_scalar(ef), op=MPI.SUM))
    
    uxe = np.pi*np.cos(np.pi*YY.ravel())*np.sin(np.pi*XX.ravel())
    uye = -np.pi*np.cos(np.pi*XX.ravel())*np.sin(np.pi*YY.ravel())
    vme = np.sqrt(uxe**2+uye**2).reshape(ny_out,nx_out)
    ge = np.nanmax(np.abs(u_grid-vme)); gl = np.sqrt(np.nanmean((u_grid-vme)**2))
    
    elapsed = time.time()-start_time
    print(f"N={N},deg={degree_u},L2={eg:.6e},Linf_g={ge:.6e},L2_g={gl:.6e},t={elapsed:.2f}s")
    
    return {"u": u_grid, "solver_info": {"mesh_resolution":N,"element_degree":degree_u,"ksp_type":"preonly","pc_type":"lu","rtol":1e-10,"iterations":nc+1,"nonlinear_iterations":[nc]}}

if __name__ == "__main__":
    cs = {"pde":{"viscosity":0.1},"output":{"grid":{"nx":100,"ny":100,"bbox":[0.0,1.0,0.0,1.0]},"field":"velocity_magnitude"}}
    r = solve(cs)
    print(f"shape={r['u'].shape}, max={np.nanmax(r['u']):.6f}, nan={np.isnan(r['u']).sum()}")
