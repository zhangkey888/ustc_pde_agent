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

    N = 64
    degree = 2
    dt_val = 0.01
    t_end = 0.1
    eps_val = 0.1
    bx, by = 1.0, 0.5

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, PETSc.ScalarType(0.0))
    pi = ufl.pi
    u_exact = ufl.exp(-t_c)*ufl.sin(pi*x[0])*ufl.sin(pi*x[1])
    # f = du/dt - eps*lap(u) + beta.grad(u)
    dudt = -ufl.exp(-t_c)*ufl.sin(pi*x[0])*ufl.sin(pi*x[1])
    lap = -2*pi*pi*ufl.exp(-t_c)*ufl.sin(pi*x[0])*ufl.sin(pi*x[1])
    dudx = ufl.exp(-t_c)*pi*ufl.cos(pi*x[0])*ufl.sin(pi*x[1])
    dudy = ufl.exp(-t_c)*pi*ufl.sin(pi*x[0])*ufl.cos(pi*x[1])
    f_expr = dudt - eps_val*lap + bx*dudx + by*dudy

    # BC
    u_bc = fem.Function(V)
    def bc_expr(t):
        return lambda xx: np.exp(-t)*np.sin(np.pi*xx[0])*np.sin(np.pi*xx[1])
    u_bc.interpolate(bc_expr(0.0))

    fdim = domain.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(u_bc, bdofs)

    # IC
    u_n = fem.Function(V)
    u_n.interpolate(lambda xx: np.sin(np.pi*xx[0])*np.sin(np.pi*xx[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    beta = ufl.as_vector([bx, by])

    a = (u*v + dt_c*eps_val*ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_c*ufl.inner(beta, ufl.grad(u))*v)*ufl.dx
    L = (u_n + dt_c*f_expr)*v*ufl.dx

    n_steps = int(round(t_end/dt_val))
    problem = petsc.LinearProblem(a, L, bcs=[bc], u=fem.Function(V),
                                  petsc_options={"ksp_type":"gmres","pc_type":"ilu","ksp_rtol":1e-10},
                                  petsc_options_prefix="cd_")
    total_iters = 0
    t = 0.0
    for step in range(n_steps):
        t += dt_val
        t_c.value = t
        u_bc.interpolate(bc_expr(t))
        uh = problem.solve()
        total_iters += problem.solver.getIterationNumber()
        u_n.x.array[:] = uh.x.array

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = []
    pts_ok = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l)>0:
            cells.append(l[0]); pts_ok.append(pts[i]); idx.append(i)
    vals = np.zeros(pts.shape[0])
    v_eval = u_n.eval(np.array(pts_ok), np.array(cells, dtype=np.int32)).flatten()
    vals[idx] = v_eval
    u_grid = vals.reshape(ny_out, nx_out)

    # initial
    XXr, YYr = np.meshgrid(xs, ys)
    u_init = np.sin(np.pi*XXr)*np.sin(np.pi*YYr)

    return {
        "u": u_grid,
        "u_initial": u_init,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type":"gmres","pc_type":"ilu","rtol":1e-10,
            "iterations": total_iters,
            "dt": dt_val, "n_steps": n_steps, "time_scheme":"backward_euler"
        }
    }
