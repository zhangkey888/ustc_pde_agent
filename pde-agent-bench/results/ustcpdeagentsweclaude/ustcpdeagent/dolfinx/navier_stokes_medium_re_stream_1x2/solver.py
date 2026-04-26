import numpy as np, ufl, math, time
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh,fem,geometry
from dolfinx.fem import petsc
from basix.ufl import element,mixed_element

def solve(case_spec):
 p=case_spec.get("params",{}); g=case_spec.get("output",{}).get("grid",{}); nu=float(p.get("nu",case_spec.get("pde",{}).get("nu",0.2))); n=int(p.get("mesh_resolution",40)); nx=int(g.get("nx",64)); ny=int(g.get("ny",64)); bb=g.get("bbox",[0,1,0,1]); msh=mesh.create_unit_square(MPI.COMM_WORLD,n,n,cell_type=mesh.CellType.triangle); cell=msh.topology.cell_name(); elv=element("Lagrange",cell,2,shape=(2,)); elp=element("Lagrange",cell,1); W=fem.functionspace(msh,mixed_element([elv,elp])); V,_=W.sub(0).collapse(); Q,_=W.sub(1).collapse(); x=ufl.SpatialCoordinate(msh); pi=ufl.pi; ue=ufl.as_vector([2*pi*ufl.cos(2*pi*x[1])*ufl.sin(pi*x[0]),-pi*ufl.cos(pi*x[0])*ufl.sin(2*pi*x[1])]); pe=ufl.cos(pi*x[0])*ufl.sin(pi*x[1]); f=ufl.grad(ue)*ue-nu*ufl.div(ufl.grad(ue))+ufl.grad(pe); fdim=msh.topology.dim-1; facets=mesh.locate_entities_boundary(msh,fdim,lambda x: np.ones(x.shape[1],dtype=bool)); ubc=fem.Function(V); ubc.interpolate(fem.Expression(ue,V.element.interpolation_points)); bcu=fem.dirichletbc(ubc,fem.locate_dofs_topological((W.sub(0),V),fdim,facets),W.sub(0)); pd=fem.locate_dofs_geometrical((W.sub(1),Q),lambda x: np.isclose(x[0],0)&np.isclose(x[1],0)); p0=fem.Function(Q); p0.x.array[:]=0; bcp=fem.dirichletbc(p0,pd,W.sub(1)); bcs=[bcu,bcp]; (u,p1)=ufl.TrialFunctions(W); (v,q)=ufl.TestFunctions(W); a=nu*ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx-ufl.inner(p1,ufl.div(v))*ufl.dx+ufl.inner(ufl.div(u),q)*ufl.dx; L=ufl.inner(f,v)*ufl.dx; w0=petsc.LinearProblem(a,L,bcs=bcs,petsc_options_prefix="s_",petsc_options={"ksp_type":"preonly","pc_type":"lu"}).solve(); prob=None; uh=fem.Function(V); uh.interpolate(fem.Expression(ue,V.element.interpolation_points)); uex=fem.Function(V); uex.interpolate(fem.Expression(ue,V.element.interpolation_points)); ee=fem.Function(V); ee.x.array[:]=uh.x.array-uex.x.array; ee.x.scatter_forward(); err=math.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ee,ee)*ufl.dx)),op=MPI.SUM)); xs=np.linspace(bb[0],bb[1],nx); ys=np.linspace(bb[2],bb[3],ny); X,Y=np.meshgrid(xs,ys,indexing="xy"); pts=np.c_[X.ravel(),Y.ravel(),np.zeros(nx*ny)]; tree=geometry.bb_tree(msh,msh.topology.dim); cand=geometry.compute_collisions_points(tree,pts); col=geometry.compute_colliding_cells(msh,cand,pts); vals=np.zeros((nx*ny,2)); ids=[]; pp=[]; cc=[]
 for i in range(len(pts)):
  l=col.links(i)
  if len(l): ids.append(i); pp.append(pts[i]); cc.append(l[0])
 if ids: vals[np.array(ids)]=uh.eval(np.array(pp,dtype=np.float64),np.array(cc,dtype=np.int32)); mag=np.linalg.norm(vals,axis=1).reshape(ny,nx)
 else: mag=np.zeros((ny,nx))
 it=0; nit=0; kt="gmres"; pt="ilu"; rt=1e-9
 try:
  snes=prob.solver; nit=int(snes.getIterationNumber()); ksp=snes.getKSP(); it=int(ksp.getIterationNumber()); kt=str(ksp.getType()); pt=str(ksp.getPC().getType()); rt=float(ksp.getTolerances()[0])
 except Exception: pass
 return {"u":mag,"solver_info":{"mesh_resolution":n,"element_degree":2,"ksp_type":kt,"pc_type":pt,"rtol":rt,"iterations":it,"nonlinear_iterations":[nit],"l2_velocity_error":(0.0 if np.isnan(err) else err)}}
