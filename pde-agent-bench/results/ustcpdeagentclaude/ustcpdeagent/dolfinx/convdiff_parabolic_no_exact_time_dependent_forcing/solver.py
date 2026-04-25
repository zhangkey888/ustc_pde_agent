import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.02
    beta_vec = np.array([6.0, 3.0])
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.001  # refined
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 192
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Boundary conditions (zero Dirichlet)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Save initial for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Variables
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta = fem.Constant(domain, PETSc.ScalarType((beta_vec[0], beta_vec[1])))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    t_c = fem.Constant(domain, PETSc.ScalarType(t0 + dt_val))

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-150.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2)) * ufl.exp(-t_c)

    # SUPG stabilization
    # tau = ( (2/dt)^2 + (2|b|/h)^2 + (4*eps/h^2)^2 )^(-1/2)
    h = ufl.CellDiameter(domain)
    b_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * b_norm / h) ** 2 + (4.0 * eps_c / h ** 2) ** 2)

    # Residuals for SUPG (backward Euler)
    # R(u) = (u - u_n)/dt - eps*div(grad(u)) + beta·grad(u) - f
    # For P1 on triangles, div(grad(u)) = 0 inside element
    Ru = (u - u_n) / dt_c - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    # Note: for P1, Laplacian term vanishes; skip it
    supg_test = tau * ufl.dot(beta, ufl.grad(v))

    # Galerkin weak form (backward Euler)
    a_gal = (u / dt_c) * v * ufl.dx \
            + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    # SUPG contribution
    a_supg = supg_test * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L_supg = supg_test * ((u_n / dt_c) + f_expr) * ufl.dx

    a_form = fem.form(a_gal + a_supg)
    L_form = fem.form(L_gal + L_supg)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    pc = solver.getPC()
    pc.setType("ilu")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

    u_sol = fem.Function(V)

    total_iters = 0
    t = t0
    for step in range(n_steps):
        t += dt_val
        t_c.value = t

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

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
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial
    u_init_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals_i = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_i.flatten()
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02}}
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"Shape: {result['u'].shape}")
    print(f"u range: [{result['u'].min():.4e}, {result['u'].max():.4e}]")
    print(f"Solver info: {result['solver_info']}")
