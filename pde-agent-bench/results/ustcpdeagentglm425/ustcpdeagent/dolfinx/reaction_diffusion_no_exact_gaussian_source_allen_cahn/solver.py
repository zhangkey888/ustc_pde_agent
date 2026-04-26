import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """
    Solve Allen-Cahn reaction-diffusion equation using IMEX scheme:
    - Implicit: backward Euler for diffusion
    - Explicit: reaction term u^3 - u
    LHS matrix is constant (assembled/factored once), only RHS changes per step.
    """
    pde = case_spec.get("pde", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})

    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])

    time_spec = pde.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.25)

    epsilon = pde.get("epsilon", 0.01)

    # Solver parameters
    mesh_resolution = 128
    element_degree = 2
    dt_actual = 0.0005
    ksp_rtol_val = 1e-10

    n_steps = int(round((t_end - t0) / dt_actual))
    if n_steps < 1:
        n_steps = 1
        dt_actual = (t_end - t0)

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                      cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = fem.Function(V)
    u_old = fem.Function(V)
    v = ufl.TestFunction(V)

    def u0_func(x):
        return 0.1 * np.exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

    u.interpolate(u0_func)
    u.x.scatter_forward()
    u_old.interpolate(u0_func)
    u_old.x.scatter_forward()

    def f_func(x):
        return 5.0 * np.exp(-180.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))

    f = fem.Function(V)
    f.interpolate(f_func)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    # IMEX: implicit diffusion + explicit reaction
    # (u - u_old)/dt - eps*laplacian(u) + (u_old^3 - u_old) = f
    # => u/dt - eps*laplacian(u) = u_old/dt - (u_old^3 - u_old) + f
    u_trial = ufl.TrialFunction(V)
    a = (u_trial * v / dt_actual
         + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v))) * ufl.dx
    L = (u_old * v / dt_actual
         - (u_old**3 - u_old) * v
         + f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble and LU-factorize ONCE
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setTolerances(rtol=ksp_rtol_val)

    b = A.createVecRight()

    total_linear_iterations = 0
    nonlinear_iterations_list = []

    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        ksp_its = solver.getIterationNumber()
        total_linear_iterations += ksp_its
        nonlinear_iterations_list.append(1)

        u_old.x.array[:] = u.x.array[:]
        u_old.x.scatter_forward()

    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    u_vals = np.zeros(nx_out * ny_out)
    pts_proc = []
    cells_proc = []
    emap = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_proc.append(pts[i])
            cells_proc.append(links[0])
            emap.append(i)

    if len(pts_proc) > 0:
        vals = u.eval(np.array(pts_proc), np.array(cells_proc, dtype=np.int32))
        u_vals[emap] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    # Sample initial condition
    u_init = fem.Function(V)
    u_init.interpolate(u0_func)
    u_init.x.scatter_forward()

    u_init_vals = np.zeros(nx_out * ny_out)
    if len(pts_proc) > 0:
        vals_init = u_init.eval(np.array(pts_proc), np.array(cells_proc, dtype=np.int32))
        u_init_vals[emap] = vals_init.flatten()

    u_init_grid = u_init_vals.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": ksp_rtol_val,
        "iterations": total_linear_iterations,
        "dt": dt_actual,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations_list,
    }

    return {"u": u_grid, "u_initial": u_init_grid, "solver_info": solver_info}
