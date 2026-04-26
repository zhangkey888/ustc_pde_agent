import math,time,numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh,fem,geometry
from dolfinx.fem import petsc as fp
import ufl
ScalarType=PETSc.ScalarType
def _kappa(expr,x):
    return eval(expr,{"__builtins__":{}},{"x":x[0],"y":x[1],"z":0*x[0],"pi":ufl.pi,"exp":ufl.exp,"sin":ufl.sin,"cos":ufl.cos,"sqrt":ufl.sqrt})
def _uex(x,t): return ufl.exp(-t)*ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
def _probe(u,pts):
    msh=u.function_space.mesh; tree=geometry.bb_tree(msh,msh.topology.dim)
    cand=geometry.compute_collisions_points(tree,pts); coll=geometry.compute_colliding_cells(msh,cand,pts)
    p=[]; c=[]; ids=[]; vals=np.full(pts.shape[0],np.nan)
    for i in range(pts.shape[0]):
        links=coll.links(i)
        if len(links)>0: p.append(pts[i]); c.append(links[0]); ids.append(i)
    if p:
        vals[np.array(ids,dtype=np.int32)]=np.asarray(u.eval(np.array(p,dtype=np.float64),np.array(c,dtype=np.int32))).reshape(-1)
    return vals
def _sample(u,grid):
    nx,ny=int(grid["nx"]),int(grid["ny"]); b=grid["bbox"]
    xs=np.linspace(b[0],b[1],nx); ys=np.linspace(b[2],b[3],ny); xx,yy=np.meshgrid(xs,ys,indexing="xy")
    out=_probe(u,np.c_[xx.ravel(),yy.ravel(),np.zeros(nx*ny)]).reshape(ny,nx)
    if np.isnan(out).any(): raise RuntimeError("sampling failed")
    return out
def _run(case,nx,deg,dt,ksp,pc,rtol):
    comm=MPI.COMM_WORLD; t0=float(case["pde"]["time"].get("t0",0.0)); t1=float(case["pde"]["time"].get("t_end",0.1))
    n=max(1,int(round((t1-t0)/dt))); dt=(t1-t0)/n
    msh=mesh.create_unit_square(comm,nx,nx,cell_type=mesh.CellType.triangle); V=fem.functionspace(msh,("Lagrange",deg)); x=ufl.SpatialCoordinate(msh)
    kap=_kappa(case["pde"]["coefficients"]["kappa"]["expr"],x)
    u_n=fem.Function(V); u_n.interpolate(fem.Expression(_uex(x,t0),V.element.interpolation_points)); u0=fem.Function(V); u0.x.array[:]=u_n.x.array[:]
    fdim=msh.topology.dim-1; facets=mesh.locate_entities_boundary(msh,fdim,lambda X: np.ones(X.shape[1],dtype=bool)); dofs=fem.locate_dofs_topological(V,fdim,facets)
    g=fem.Function(V); g.interpolate(fem.Expression(_uex(x,t0+dt),V.element.interpolation_points)); bc=fem.dirichletbc(g,dofs)
    ff=fem.Function(V); ff.interpolate(fem.Expression(-_uex(x,t0+dt)-ufl.div(kap*ufl.grad(_uex(x,t0+dt))),V.element.interpolation_points))
    u,v=ufl.TrialFunction(V),ufl.TestFunction(V)
    a=fem.form((u*v+dt*ufl.inner(kap*ufl.grad(u),ufl.grad(v)))*ufl.dx); L=fem.form((u_n*v+dt*ff*v)*ufl.dx)
    A=fp.assemble_matrix(a,bcs=[bc]); A.assemble(); b=fp.create_vector(L.function_spaces)
    solver=PETSc.KSP().create(comm); solver.setOperators(A); solver.setType(ksp); solver.getPC().setType(pc); solver.setTolerances(rtol=rtol)
    uh=fem.Function(V); ittot=0; tic=time.perf_counter()
    for i in range(n):
        t=t0+(i+1)*dt
        g.interpolate(fem.Expression(_uex(x,t),V.element.interpolation_points))
        ff.interpolate(fem.Expression(-_uex(x,t)-ufl.div(kap*ufl.grad(_uex(x,t))),V.element.interpolation_points))
        with b.localForm() as loc: loc.set(0)
        fp.assemble_vector(b,L); fp.apply_lifting(b,[a],bcs=[[bc]]); b.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE); fp.set_bc(b,[bc])
        solver.solve(b,uh.x.petsc_vec); uh.x.scatter_forward(); it=solver.getIterationNumber(); ittot+=max(it,0); u_n.x.array[:]=uh.x.array[:]
    elapsed=time.perf_counter()-tic
    ueT=fem.Function(V); ueT.interpolate(fem.Expression(_uex(x,t1),V.element.interpolation_points)); e=fem.Function(V); e.x.array[:]=uh.x.array-ueT.x.array
    err=math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e,e)*ufl.dx)),op=MPI.SUM))
    ref=math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ueT,ueT)*ufl.dx)),op=MPI.SUM))
    return {"u_h":uh,"u0":u0,"mesh_resolution":nx,"element_degree":deg,"dt":dt,"n_steps":n,"time_scheme":"backward_euler","ksp_type":solver.getType(),"pc_type":solver.getPC().getType(),"rtol":rtol,"iterations":int(ittot),"l2_error":err,"rel_l2_error":err/max(ref,1e-16),"wall":elapsed}
def solve(case_spec:dict)->dict:
    budget=7.286; start=time.perf_counter(); best=None
    for nx,deg,dt in [(40,1,0.01),(56,1,0.0075),(72,1,0.005),(88,1,0.004)]:
        if time.perf_counter()-start>budget*0.82 and best is not None: break
        try: cur=_run(case_spec,nx,deg,dt,"cg","hypre",1e-10)
        except Exception: cur=_run(case_spec,nx,deg,dt,"preonly","lu",1e-12)
        if best is None or cur["l2_error"]<=best["l2_error"]: best=cur
        elapsed=time.perf_counter()-start
        if cur["l2_error"]<3.39e-4 and elapsed>1.5: break
        if cur["wall"]<0.6 and elapsed<budget*0.35 and cur["l2_error"]>1.0e-4: continue
        if elapsed>budget*0.55: break
    return {"u":_sample(best["u_h"],case_spec["output"]["grid"]),"u_initial":_sample(best["u0"],case_spec["output"]["grid"]),"solver_info":{"mesh_resolution":int(best["mesh_resolution"]),"element_degree":int(best["element_degree"]),"ksp_type":str(best["ksp_type"]),"pc_type":str(best["pc_type"]),"rtol":float(best["rtol"]),"iterations":int(best["iterations"]),"dt":float(best["dt"]),"n_steps":int(best["n_steps"]),"time_scheme":str(best["time_scheme"]),"l2_error":float(best["l2_error"]),"rel_l2_error":float(best["rel_l2_error"])}} 
if __name__=="__main__":
    case={"pde":{"time":{"t0":0.0,"t_end":0.1,"dt":0.01},"coefficients":{"kappa":{"type":"expr","expr":"0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}}},"output":{"grid":{"nx":32,"ny":32,"bbox":[0.0,1.0,0.0,1.0]}}}
    out=solve(case)
    if MPI.COMM_WORLD.rank==0: print(out["u"].shape,out["solver_info"])
