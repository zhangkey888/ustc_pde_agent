import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.5)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")

    # Parameters
    nx = ny = 64
    degree = 2
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly

    epsilon = 1.0  # diffusion coefficient (default for this problem)
    # Check if epsilon is specified
    if "coefficients" in pde:
        epsilon = pde["coefficients"].get("epsilon", 1.0)

    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))

    # Manufactured solution: u = exp(-t) * sin(pi*x) * sin(pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Reaction term R(u) = u (linear reaction)
    # For the manufactured solution:
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -eps * nabla^2 u = eps * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # R(u) = u = exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - eps*nabla^2 u + R(u)
    #   = -exp(-t)*sin + eps*2*pi^2*exp(-t)*sin + exp(-t)*sin
    #   = exp(-t)*sin*(2*eps*pi^2)
    # But let's compute it symbolically to be safe

    # Source term derived from manufactured solution
    # du/dt - eps*laplacian(u) + R(u) = f
    # R(u) = u (linear reaction)
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) + eps*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y) + exp(-t)*sin(pi*x)*sin(pi*y)
    # f = exp(-t)*sin(pi*x)*sin(pi*y) * (2*eps*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (2.0 * epsilon * ufl.pi**2)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")

    # Current solution
    u_h = fem.Function(V, name="u_h")

    # Time step constant
    dt_c = fem.Constant(domain, ScalarType(dt))

    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + R(u) = f
    # Weak form: (u - u_n)/dt * v dx + eps * grad(u) . grad(v) dx + u * v dx = f * v dx
    # Linear reaction R(u) = u
    a = (u * v / dt_c) * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_ufl * v * ufl.dx

    # Boundary conditions: u = g on boundary
    # g = exp(-t)*sin(pi*x)*sin(pi*y) = 0 on boundary of unit square
    # So homogeneous Dirichlet BC
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # BC function (will be updated each time step)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0  # zero on boundary for this problem
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]

    # Set initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    # Create RHS vector
    b = fem.petsc.create_vector(L_form)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()

    total_iterations = 0

    # Time stepping
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt
        t.value = current_t

        # Update BC if needed (stays zero for this problem)
        # u_bc is already zero

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate solution on output grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition
    u_n_init = fem.Function(V)
    u_n_init.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    u_init_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()

    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }