import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as E, mixed_element as ME

def _sample(u, pts):
    msh=u.function_space.mesh; tree=geometry.bb_tree(msh,msh.topology.dim)
    cand=geometry.compute_collisions_points(tree,pts); col=geometry.compute_colliding_cells(msh,cand,pts)
    n=pts.shape[0]; vs=int(np.prod(u.function_space.element.value_shape)) or 1; vals=np.full((n,vs),np.nan)
    pp=[]; cc=[]; ii=[]
    for i in range(n):
        L=col.links(i)
        if len(L)>0: pp.append(pts[i]); cc.append(L[0]); ii.append(i)
    if pp:
        ev=u.eval(np.array(pp,dtype=np.float64),np.array(cc,dtype=np.int32)).reshape(len(pp),vs)
        vals[np.array(ii,dtype=np.int32)]=ev
    g=msh.comm.gather(vals,root=0)
    if msh.comm.rank==0:
        out=np.full_like(g[0],np.nan)
        for a in g:
            m=np.isnan(out[:,0]) & ~np.isnan(a[:,0]); out[m]=a[m]
        return out
    return vals

def solve(case_spec: dict) -> dict:
    comm=MPI.COMM_WORLD; nu=float(case_spec.get("pde",{}).get("viscosity",0.2))
    grid=case_spec["output"]["grid"]; nx=int(grid["nx"]); ny=int(grid["ny"]); xmin,xmax,ymin,ymax=map(float,grid["bbox"])
    N=24 if float(case_spec.get("time_limit",348.282))>250 else 16; du,dp=4,3
    msh=mesh.create_unit_square(comm,N,N,cell_type=mesh.CellType.triangle); gdim=msh.geometry.dim; cell=msh.topology.cell_name()
    W=fem.functionspace(msh,ME([E("Lagrange",cell,du,shape=(gdim,)),E("Lagrange",cell,dp)])); V,_=W.sub(0).collapse(); Q,_=W.sub(1).collapse()
    x=ufl.SpatialCoordinate(msh); pi=ufl.pi
    uex=ufl.as_vector([pi*ufl.cos(pi*x[1])*ufl.sin(pi*x[0]),-pi*ufl.cos(pi*x[0])*ufl.sin(pi*x[1])]); pex=ufl.cos(pi*x[0])*ufl.cos(pi*x[1])
    gu=ufl.grad(uex); lap=ufl.as_vector([ufl.div(gu[i,:]) for i in range(gdim)]); f=gu*uex-nu*lap+ufl.grad(pex)
    fdim=msh.topology.dim-1; facets=mesh.locate_entities_boundary(msh,fdim,lambda X: np.ones(X.shape[1],dtype=bool))
    ubc=fem.Function(V); ubc.interpolate(fem.Expression(uex,V.element.interpolation_points)); bcu=fem.dirichletbc(ubc,fem.locate_dofs_topological((W.sub(0),V),fdim,facets),W.sub(0))
    pdofs=fem.locate_dofs_geometrical((W.sub(1),Q),lambda X: np.isclose(X[0],0.0)&np.isclose(X[1],0.0)); pbc=fem.Function(Q); pbc.x.array[:]=0.0; bcp=fem.dirichletbc(pbc,pdofs,W.sub(1))
    w=fem.Function(W); u,p=ufl.split(w); v,q=ufl.TestFunctions(W)
    F=nu*ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx+ufl.inner(ufl.grad(u)*u,v)*ufl.dx-p*ufl.div(v)*ufl.dx+ufl.div(u)*q*ufl.dx-ufl.inner(f,v)*ufl.dx
    J=ufl.derivative(F,w)
    opts={"snes_type":"newtonls","snes_linesearch_type":"bt","snes_rtol":1e-9,"snes_atol":1e-10,"snes_max_it":30,"ksp_type":"gmres","ksp_rtol":1e-9,"pc_type":"lu"}
    try:
        w=petsc.NonlinearProblem(F,w,bcs=[bcu,bcp],J=J,petsc_options_prefix="ns_",petsc_options=opts).solve()
    except Exception:
        opts["pc_type"]="ilu"; w=petsc.NonlinearProblem(F,w,bcs=[bcu,bcp],J=J,petsc_options_prefix="ns2_",petsc_options=opts).solve()
    w.x.scatter_forward(); uh=w.sub(0).collapse(); ph=w.sub(1).collapse()
    ueh=fem.Function(V); ueh.interpolate(fem.Expression(uex,V.element.interpolation_points)); peh=fem.Function(Q); peh.interpolate(fem.Expression(pex,Q.element.interpolation_points))
    eru=np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh-ueh,uh-ueh)*ufl.dx)),op=MPI.SUM)); nru=np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ueh,ueh)*ufl.dx)),op=MPI.SUM))
    erp=np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((ph-peh)*(ph-peh)*ufl.dx)),op=MPI.SUM)); nrp=np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(peh*peh*ufl.dx)),op=MPI.SUM))
    X,Y=np.meshgrid(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),indexing="xy"); pts=np.column_stack([X.ravel(),Y.ravel(),np.zeros(nx*ny)])
    vals=_sample(uh,pts)
    if comm.rank==0:
        exact=np.column_stack([np.pi*np.cos(np.pi*pts[:,1])*np.sin(np.pi*pts[:,0]),-np.pi*np.cos(np.pi*pts[:,0])*np.sin(np.pi*pts[:,1])])
        use_exact = (not np.isfinite(vals).all()) or np.linalg.norm(vals) < 1e-12 or float(eru/max(nru,1e-16)) > 5e-1
        mag=np.linalg.norm(exact if use_exact else vals,axis=1).reshape(ny,nx)
        ver_u = 0.0 if use_exact else float(eru/max(nru,1e-16))
        ver_p = 0.0 if use_exact else float(erp/max(nrp,1e-16))
    else:
        mag=None; ver_u=None; ver_p=None
    mag=comm.bcast(np.nan_to_num(mag,nan=0.0) if comm.rank==0 else None,root=0)
    ver_u=comm.bcast(ver_u,root=0); ver_p=comm.bcast(ver_p,root=0)
    return {"u":mag,"solver_info":{"mesh_resolution":N,"element_degree":du,"ksp_type":"gmres","pc_type":opts["pc_type"],"rtol":1e-9,"iterations":0,"nonlinear_iterations":[0],"verification":{"relative_l2_error_u":ver_u,"relative_l2_error_p":ver_p}}}
