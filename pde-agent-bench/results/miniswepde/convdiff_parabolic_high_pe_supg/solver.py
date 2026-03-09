import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve transient convection-diffusion with SUPG stabilization."""

    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})

    # Diffusion and convection parameters
    params = pde.get("parameters", {})
    epsilon_val = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [10.0, 4.0])

    # Time parameters - hardcoded defaults as fallback
    time_spec = pde.get("time", {})
    t_end = time_spec.get("t_end", 0.08)
    dt_val = time_spec.get("dt", 0.01)
    scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True  # forced for this problem

    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # ---- Solve at chosen resolution ----
    N = 64
    element_degree = 1

    comm = MPI.COMM_WORLD

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous time step solution
    u_n = fem.Function(V, name="u_n")

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Time as a constant (will be updated)
    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt_val))

    # Parameters
    epsilon = fem.Constant(domain, ScalarType(epsilon_val))
    beta = fem.Constant(domain, ScalarType((beta_vec[0], beta_vec[1])))

    pi = ufl.pi

    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])

    # Source term f = du/dt - eps*laplacian(u) + beta.grad(u)
    du_dt = -ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))

    f = du_dt - epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)

    # ---- SUPG Stabilization ----
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * epsilon)
    # SUPG parameter with coth formula
    tau_supg = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # ---- Backward Euler weak form ----
    # Galerkin part
    a_galerkin = (u / dt_const * v * ufl.dx
                  + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx)

    L_galerkin = (u_n / dt_const * v * ufl.dx
                  + f * v * ufl.dx)

    # SUPG stabilization: test function modification v_supg = tau * beta.grad(v)
    v_supg = tau_supg * ufl.dot(beta, ufl.grad(v))

    # For P1 elements, laplacian terms vanish element-wise
    a_supg = (u / dt_const * v_supg * ufl.dx
              + ufl.dot(beta, ufl.grad(u)) * v_supg * ufl.dx)

    L_supg = (u_n / dt_const * v_supg * ufl.dx
              + f * v_supg * ufl.dx)

    # Total forms
    a_total = a_galerkin + a_supg
    L_total = L_galerkin + L_supg

    # ---- Boundary conditions ----
    u_bc_func = fem.Function(V)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]

    # ---- Initial condition ----
    u_n.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Store initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # ---- Compile forms ----
    a_form = fem.form(a_total)
    L_form = fem.form(L_total)

    # ---- Solver setup ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(V)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)

    # Solution function
    u_sol = fem.Function(V)

    # ---- Time stepping ----
    t = 0.0
    n_steps = int(np.ceil(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_const.value = actual_dt

    total_iterations = 0

    for step in range(n_steps):
        t += actual_dt
        t_const.value = t

        # Update boundary condition
        t_current = t
        u_bc_func.interpolate(
            lambda x, tc=t_current: np.exp(-tc) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
        )

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]

    # ---- Evaluate on output grid ----
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    u_initial_grid = _evaluate_on_grid(domain, u_initial_func, nx_out, ny_out)

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

    return result


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate a dolfinx Function on a uniform nx x ny grid over [0,1]^2."""

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_values = np.full(points_2d.shape[0], np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_2d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx, ny))
