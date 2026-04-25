import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    mesh_res = 160
    elem_deg = 3
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    kappa = 1.0
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_ufl = ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx
    fdim = domain.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "pc_hypre_type": "boomeramg", "ksp_rtol": rtol, "ksp_atol": 1e-12, "ksp_max_it": 500}, petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    ksp = problem.solver
    iters = ksp.getIterationNumber()
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    bb = geometry.bb_tree(domain, domain.topology.dim)
    cc = geometry.compute_collisions_points(bb, pts.T)
    colcells = geometry.compute_colliding_cells(domain, cc, pts.T)
    ponp, conp, emap = [], [], []
    for i in range(pts.shape[1]):
        lk = colcells.links(i)
        if len(lk) > 0:
            ponp.append(pts.T[i])
            conp.append(lk[0])
            emap.append(i)
    uvals = np.zeros(pts.shape[1])
    if len(ponp) > 0:
        vs = u_sol.eval(np.array(ponp), np.array(conp, dtype=np.int32))
        uvals[emap] = vs.flatten()
    ugrid = uvals.reshape(ny_out, nx_out)
    if comm.size > 1:
        ug = np.zeros_like(ugrid)
        comm.Reduce(ugrid, ug, op=MPI.SUM, root=0)
        ugrid = ug if comm.rank == 0 else ugrid
    si = {"mesh_resolution": mesh_res, "element_degree": elem_deg, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iters}
    return {"u": ugrid, "solver_info": si}
