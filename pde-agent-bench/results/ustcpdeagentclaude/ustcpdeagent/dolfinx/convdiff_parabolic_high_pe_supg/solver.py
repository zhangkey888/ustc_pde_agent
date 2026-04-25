import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps_val = 0.01
    beta_val = np.array([10.0, 4.0])
    t0 = 0.0
    t_end = 0.08
    dt_val = 0.01

    # Grid output spec
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh
    N = 96
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(0.0))

    # Exact solution: u = exp(-t) sin(2pi x) sin(pi y)
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    # Source f = du/dt - eps*lap(u) + beta . grad(u)
    # du/dt = -u
    # lap(u) = -(4pi^2 + pi^2) u = -5 pi^2 u
    # grad(u) = exp(-t)*(2pi cos(2pi x) sin(pi y), pi sin(2pi x) cos(pi y))
    du_dt = -u_exact_expr
    lap_u = -(4*ufl.pi**2 + ufl.pi**2) * u_exact_expr
    grad_u = ufl.as_vector([
        ufl.exp(-t_const) * 2*ufl.pi*ufl.cos(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]),
        ufl.exp(-t_const) * ufl.pi*ufl.sin(2*ufl.pi*x[0])*ufl.cos(ufl.pi*x[1]),
    ])
    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))

    f_expr = du_dt - eps_c * lap_u + ufl.dot(beta, grad_u)

    # Boundary condition (time-dependent)
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    u0_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    # Variational form - Backward Euler with SUPG
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))

    # Standard Galerkin: (u - u_n)/dt + beta.grad(u) - eps*lap(u) = f
    # Weak form: (u,v)/dt + eps*(grad u, grad v) + (beta.grad u, v) = (u_n,v)/dt + (f,v)
    a_gal = (u*v/dt_c + eps_c*ufl.dot(ufl.grad(u), ufl.grad(v))
             + ufl.dot(beta, ufl.grad(u))*v) * ufl.dx
    L_gal = (u_n*v/dt_c + f_expr*v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    # tau for transient conv-diff
    Pe_h = beta_norm * h / (2*eps_c)
    tau = h / (2*beta_norm) * (1.0 / ufl.tanh(Pe_h) - 1.0/Pe_h)

    # residual (strong form) of trial
    r_trial = u/dt_c + ufl.dot(beta, ufl.grad(u)) - eps_c*ufl.div(ufl.grad(u))
    r_rhs = u_n/dt_c + f_expr

    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    a_supg = r_trial * supg_test * ufl.dx
    L_supg = r_rhs * supg_test * ufl.dx

    a_form = fem.form(a_gal + a_supg)
    L_form = fem.form(L_gal + L_supg)

    # Assemble matrix once (time-independent if dt fixed)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)

    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0

    t_cur = t0
    for step in range(n_steps):
        t_cur += dt_val
        t_const.value = t_cur
        # Update BC
        u_bc.interpolate(u_bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_h.x.array[:]

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cell_cands, pts)
    cells = []
    pts_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)
    vals = np.zeros(pts.shape[0])
    if len(pts_on) > 0:
        ev = u_h.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        vals[idx_map] = ev.flatten()
    u_grid = vals.reshape(ny_out, nx_out)

    # Initial condition grid
    u_n_init = fem.Function(V)
    t_const.value = t0
    u_n_init.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    vals0 = np.zeros(pts.shape[0])
    if len(pts_on) > 0:
        ev0 = u_n_init.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        vals0[idx_map] = ev0.flatten()
    u_init_grid = vals0.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    u = res["u"]
    # Error vs exact
    xs = np.linspace(0, 1, 128)
    ys = np.linspace(0, 1, 128)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.08) * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u - u_ex)**2))
    maxerr = np.max(np.abs(u - u_ex))
    print(f"Wall time: {elapsed:.3f}s")
    print(f"RMS error: {err:.3e}")
    print(f"Max error: {maxerr:.3e}")
    print(f"Iterations: {res['solver_info']['iterations']}")
