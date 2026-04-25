import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    k = 18.0

    N = 128
    degree = 3
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.tanh(6*(x[0]-0.5)) * ufl.sin(ufl.pi*x[1])
    f_expr = -ufl.div(ufl.grad(u_exact)) - k*k*u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k*k*ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, bdofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type":"preonly","pc_type":"lu",
                                                  "pc_factor_mat_solver_type":"mumps"},
                                   petsc_options_prefix="hh_")
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    pop, cop, emap = [], [], []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            pop.append(pts[i]); cop.append(l[0]); emap.append(i)
    vals = np.full(pts.shape[0], np.nan)
    v_ev = u_sol.eval(np.array(pop), np.array(cop, dtype=np.int32))
    vals[emap] = v_ev.flatten()
    u_grid = vals.reshape(ny_out, nx_out)

    # error check
    XXf = XX.ravel(); YYf = YY.ravel()
    u_ex = np.tanh(6*(XXf-0.5))*np.sin(np.pi*YYf)
    err = np.sqrt(np.mean((vals - u_ex)**2))
    print(f"RMS err = {err:.3e}")

    return {"u": u_grid,
            "solver_info":{"mesh_resolution":N,"element_degree":degree,
                           "ksp_type":"preonly","pc_type":"lu","rtol":1e-12,
                           "iterations":int(its)}}

if __name__=="__main__":
    import time
    spec = {"output":{"grid":{"nx":128,"ny":128,"bbox":[0,1,0,1]}}}
    t=time.time(); r=solve(spec); print("time",time.time()-t)
