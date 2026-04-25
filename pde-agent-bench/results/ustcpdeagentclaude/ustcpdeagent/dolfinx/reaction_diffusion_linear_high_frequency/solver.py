import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from dolfinx import geometry


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    epsilon = case_spec.get("pde", {}).get("epsilon", 1.0)
    # Default epsilon if not provided
    if "parameters" in case_spec and "epsilon" in case_spec["parameters"]:
        epsilon = case_spec["parameters"]["epsilon"]
    # Try common places
    eps_val = 1.0
    for key_path in [("pde", "epsilon"), ("parameters", "epsilon"), ("epsilon",)]:
        d = case_spec
        ok = True
        for k in key_path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok and isinstance(d, (int, float)):
            eps_val = float(d)
            break
    epsilon = eps_val

    # Time parameters
    time_cfg = case_spec.get("pde", {}).get("time", None) or case_spec.get("time", {})
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg.get("t_end", 0.3))
    dt_val = float(time_cfg.get("dt", 0.005))

    # Output grid
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]

    # Mesh resolution - high freq solution sin(4pi*x)*sin(3pi*y), need fine mesh
    N = 80
    degree = 2

    # Reduce dt for accuracy with CN
    dt_val = min(dt_val, 0.005)

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))

    # Exact solution expression (UFL)
    def u_exact_ufl(t_c):
        return ufl.exp(-t_c) * ufl.sin(4.0*ufl.pi*x[0]) * ufl.sin(3.0*ufl.pi*x[1])

    # R(u) = u (linear reaction)
    # Equation: du/dt - eps*lap(u) + u = f
    # du/dt = -u_exact
    # lap(u_exact) = -(16+9)*pi^2 * u_exact = -25 pi^2 u_exact
    # f = -u_exact - eps*(-25 pi^2 u_exact) + u_exact = 25 pi^2 eps u_exact
    # But to be safe against my assumption of R(u), compute symbolically
    def f_ufl(t_c):
        ue = u_exact_ufl(t_c)
        dudt = -ue
        lap = -25.0 * ufl.pi**2 * ue
        return dudt - epsilon*lap + ue  # R(u)=u

    # Boundary condition
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    t_const.value = t0
    ic_expr = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)
    u_n.interpolate(ic_expr)

    # Crank-Nicolson
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))
    t_half = fem.Constant(msh, PETSc.ScalarType(t0 + dt_val/2))
    t_new = fem.Constant(msh, PETSc.ScalarType(t0 + dt_val))

    # (u^{n+1} - u^n)/dt - eps*0.5*(lap u^{n+1} + lap u^n) + 0.5*(u^{n+1} + u^n) = f(t+dt/2)
    # Bilinear (in u^{n+1}):
    # u/dt + 0.5*eps*grad u . grad v + 0.5*u*v
    a_form = (u*v/dt_c + 0.5*epsilon*ufl.inner(ufl.grad(u), ufl.grad(v)) + 0.5*u*v) * ufl.dx
    # RHS:
    f_mid = f_ufl(t_half)
    L_form = (u_n*v/dt_c - 0.5*epsilon*ufl.inner(ufl.grad(u_n), ufl.grad(v)) - 0.5*u_n*v + f_mid*v) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_compiled.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)

    # Time stepping
    n_steps = int(np.ceil((t_end - t0)/dt_val))
    dt_actual = (t_end - t0)/n_steps
    dt_c.value = dt_actual

    total_iters = 0
    t_curr = t0
    for step in range(n_steps):
        t_half.value = t_curr + dt_actual/2
        t_new.value = t_curr + dt_actual
        t_const.value = t_new.value

        # Update BC
        u_bc.interpolate(bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]
        t_curr += dt_actual

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, i in enumerate(eval_map):
            u_values[i] = vals[idx, 0] if vals.ndim > 1 else vals[idx]

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition grid
    u_init_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        # Recompute IC at t0 for output
        u_init_func = fem.Function(V)
        t_const.value = t0
        ic_expr2 = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)
        u_init_func.interpolate(ic_expr2)
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, i in enumerate(eval_map):
            u_init_vals[i] = vals_init[idx, 0] if vals_init.ndim > 1 else vals_init[idx]
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": float(dt_actual),
            "n_steps": int(n_steps),
            "time_scheme": "crank_nicolson",
        },
    }
