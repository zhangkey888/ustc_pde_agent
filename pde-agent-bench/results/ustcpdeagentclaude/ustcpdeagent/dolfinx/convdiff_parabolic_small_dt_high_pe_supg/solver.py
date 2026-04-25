import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps_val = 0.01
    beta_vec = np.array([12.0, 4.0])
    t0 = 0.0
    t_end = 0.06
    dt_val = 0.005

    # Mesh
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt_val))
    eps_const = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, ScalarType((beta_vec[0], beta_vec[1])))

    # Exact solution u = exp(-t)*sin(4*pi*x)*sin(pi*y)
    def u_exact_expr(t_c):
        return ufl.exp(-t_c) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    u_ex = u_exact_expr(t_const)

    # Compute f = du/dt - eps*laplacian(u) + beta . grad(u)
    # du/dt = -u
    # laplacian: -(16 pi^2 + pi^2) u = -17 pi^2 u
    # grad u: (4pi cos(4pi x) sin(pi y), pi sin(4pi x) cos(pi y)) * exp(-t)
    u_sym = u_ex
    dudt = -u_sym
    lap = ufl.div(ufl.grad(u_sym))
    grad_u = ufl.grad(u_sym)
    f_expr = dudt - eps_const * lap + ufl.dot(beta, grad_u)

    # Initial condition
    u_n = fem.Function(V)
    u_n_expr = fem.Expression(u_exact_expr(fem.Constant(domain, ScalarType(t0))),
                               V.element.interpolation_points)
    u_n.interpolate(u_n_expr)

    # Store initial grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler:
    # (u - u_n)/dt - eps*lap(u) + beta . grad(u) = f
    # => u/dt - eps*lap(u) + beta.grad(u) = f + u_n/dt

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    # tau for transient: 1/(2/dt + 2|beta|/h + 4*eps/h^2)
    tau = 1.0 / (2.0/dt_const + 2.0*beta_norm/h + 4.0*eps_const/(h*h))

    # Galerkin part
    a_gal = (u/dt_const)*v*ufl.dx \
          + eps_const*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx \
          + ufl.dot(beta, ufl.grad(u))*v*ufl.dx

    L_gal = (u_n/dt_const)*v*ufl.dx + f_expr*v*ufl.dx

    # SUPG: test function modification: tau * beta . grad(v)
    # Residual (strong): u/dt - eps*lap(u) + beta.grad(u) - f - u_n/dt
    r_a = u/dt_const - eps_const*ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_L = f_expr + u_n/dt_const

    a_supg = tau * r_a * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_supg = tau * r_L * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a_form = a_gal + a_supg
    L_form = L_gal + L_supg

    # Update BC value at time t
    u_bc_expr_container = {}

    u_sol = fem.Function(V)

    # Assemble matrix once (doesn't depend on t for linear problem with constant coeffs)
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_compiled.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0
    t_current = t0

    # Sample initial
    def sample_on_grid(func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])

        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_cand = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(domain, cell_cand, pts)

        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(pts.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                points_on_proc.append(pts[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)

        vals = np.full(pts.shape[0], np.nan)
        if len(points_on_proc) > 0:
            v_arr = func.eval(np.array(points_on_proc),
                              np.array(cells_on_proc, dtype=np.int32))
            vals[eval_map] = v_arr.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    # Prepare BC interp expression at current time
    bc_expr_factory = lambda t_val: fem.Expression(
        ufl.exp(-t_val) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
        V.element.interpolation_points
    )

    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current

        # Update boundary condition
        u_bc.interpolate(bc_expr_factory(fem.Constant(domain, ScalarType(t_current))))

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

        # Update previous
        u_n.x.array[:] = u_sol.x.array

    # Sample final solution
    u_grid = sample_on_grid(u_n)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
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
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
        "pde": {"time": True},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.3f}s")
    u = result["u"]
    print(f"Shape: {u.shape}, min={u.min():.4f}, max={u.max():.4f}")

    # Check error
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    t_end = 0.06
    u_exact = np.exp(-t_end) * np.sin(4*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u - u_exact)**2))
    print(f"L2 error: {err:.6e}")
    print(f"Max error: {np.max(np.abs(u - u_exact)):.6e}")
