import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx_out, ny_out = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 64
    degree = 2
    kappa = 0.1
    t0, t_end, dt = 0.0, 0.2, 0.02
    n_steps = int(round((t_end - t0) / dt))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))

    # u_exact = exp(-0.5 t) sin(2pi x) sin(pi y)
    def u_exact_ufl(tc):
        return ufl.exp(-0.5*tc) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    u_ex = u_exact_ufl(t_const)
    # f = du/dt - kappa*lap(u) = (-0.5 + kappa*(4pi^2 + pi^2)) * u_exact
    f_expr = (-0.5 + kappa*(4*ufl.pi**2 + ufl.pi**2)) * u_ex

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_sol = fem.Function(V)

    # Initial condition
    t_const.value = t0
    u_init_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    u_sol.interpolate(u_init_expr)

    # Save initial for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u_init_expr)

    # Backward Euler: (u - u_n)/dt - kappa lap u = f  =>  u + dt*kappa*grad u . grad v = u_n + dt*f
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt))
    a = u*v*ufl.dx + dt_c*kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = (u_n + dt_c*f_expr)*v*ufl.dx

    # BC
    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)
    u_bc = fem.Function(V)

    problem = petsc.LinearProblem(
        a, L, bcs=[fem.dirichletbc(u_bc, bdofs)], u=u_sol,
        petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10},
        petsc_options_prefix="heat_"
    )

    total_iters = 0
    for step in range(n_steps):
        t_new = t0 + (step+1)*dt
        t_const.value = t_new
        u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
        problem.solve()
        total_iters += problem.solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cands, pts)

    cells = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        cells.append(links[0] if len(links) > 0 else 0)
    cells = np.array(cells, dtype=np.int32)

    u_vals = u_sol.eval(pts, cells).reshape(ny_out, nx_out)
    u_init_vals = u_initial_func.eval(pts, cells).reshape(ny_out, nx_out)

    return {
        "u": u_vals,
        "u_initial": u_init_vals,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt, "n_steps": n_steps, "time_scheme": "backward_euler"
        }
    }
