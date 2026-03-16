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
    N = 64
    degree = 2
    dt_val = 0.005
    t_end = 0.1
    n_steps = int(round(t_end / dt_val))

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))

    # Exact solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # kappa = 0.2 + exp(-120*((x-0.55)^2 + (y-0.45)^2))
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))

    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # We compute f symbolically
    dudt = -ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f = dudt - ufl.div(kappa * grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)

    # Initial condition
    u_n_expr = fem.Expression(
        ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    )
    t.value = 0.0
    u_n.interpolate(u_n_expr)

    # Save initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0] = XX.ravel()
    points_2d[1] = YY.ravel()

    # Build point evaluation infrastructure
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)

    def eval_on_grid(u_func):
        u_values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(points_on_proc_arr, cells_on_proc_arr)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape((nx_out, ny_out))

    u_initial = eval_on_grid(u_n)

    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f
    # Weak form: (u/dt)*v*dx + kappa*grad(u)*grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    a_form = (u / dt) * v * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (u_n / dt) * v * ufl.dx + f * v * ufl.dx

    # Boundary conditions (u = u_exact on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix (kappa doesn't depend on time, but the form has dt which is constant)
    A = petsc.assemble_matrix(a_compiled, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
    A.assemble()

    b = fem.petsc.create_vector(L_compiled)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    u_sol = fem.Function(V)

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val

        # Update BC
        u_bc.interpolate(bc_expr)
        bc = fem.dirichletbc(u_bc, boundary_dofs)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Re-assemble matrix if BCs change structure (they don't here, but we need correct BC rows)
        # Actually for Dirichlet BCs the matrix rows are set during assemble_matrix
        # Since BCs don't change location, we can reuse A. But we need to reassemble if bc values changed.
        # For backward Euler with constant kappa and dt, A doesn't change. BC rows are zeroed with 1 on diagonal.
        # The matrix was assembled once with the BC pattern - this is fine.

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate final solution on grid
    u_grid = eval_on_grid(u_sol)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }