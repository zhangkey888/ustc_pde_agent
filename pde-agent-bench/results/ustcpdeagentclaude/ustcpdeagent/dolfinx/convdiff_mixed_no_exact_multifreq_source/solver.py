import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 500
    degree = 2
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps_val = 0.01
    beta_vec = np.array([12.0, 6.0])
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta = fem.Constant(msh, PETSc.ScalarType(beta_vec))

    x = ufl.SpatialCoordinate(msh)
    f = ufl.sin(8*ufl.pi*x[0])*ufl.sin(6*ufl.pi*x[1]) + 0.3*ufl.sin(12*ufl.pi*x[0])*ufl.sin(10*ufl.pi*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin
    a = eps_c*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)*ufl.dx
    L = f*v*ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = bnorm*h/(2*eps_c)
    # tau
    tau = (h/(2*bnorm))*(1.0/ufl.tanh(Pe) - 1.0/Pe)
    # residual (strong) for trial
    res_u = -eps_c*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau*res_u*ufl.dot(beta, ufl.grad(v))*ufl.dx
    L += tau*f*ufl.dot(beta, ufl.grad(v))*ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V); u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10, "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    uh = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(XX.size)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cc = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(msh, cc, pts)
    pop, cop, idx = [], [], []
    for i in range(pts.shape[0]):
        lk = col.links(i)
        if len(lk) > 0:
            pop.append(pts[i]); cop.append(lk[0]); idx.append(i)
    vals = np.zeros(pts.shape[0])
    if pop:
        v_ = uh.eval(np.array(pop), np.array(cop, dtype=np.int32)).flatten()
        vals[idx] = v_
    u_grid = vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-10,
            "iterations": int(its),
        }
    }

if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    r = solve(spec)
    print("time", time.time()-t0, "its", r["solver_info"]["iterations"], "range", r["u"].min(), r["u"].max())
