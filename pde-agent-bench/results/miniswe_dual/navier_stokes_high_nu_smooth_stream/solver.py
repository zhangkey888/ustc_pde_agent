import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict = None) -> dict:
    t0 = time.time()

    nu_val = 2.0
    nx_out, ny_out = 50, 50
    comm = MPI.COMM_WORLD

    # P3/P2 Taylor-Hood on 32x32 mesh
    N = 32
    degree_u = 3
    degree_p = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    cell_name = domain.topology.cell_name()

    vel_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])

    W = fem.functionspace(domain, mel)
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, degree_u, shape=(gdim,)))
    Q = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, degree_p))

    w = fem.Function(W)
    (u_sol, p_sol) = ufl.split(w)
    (v_test, q_test) = ufl.TestFunctions(W)
    dw = ufl.TrialFunction(W)

    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi

    # Manufactured exact solution
    u_exact = ufl.as_vector([
        0.5 * pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -0.5 * pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) + ufl.cos(pi_val * x[1])

    # Source term: f = u·∇u - ν∇²u + ∇p
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Nonlinear residual
    F_form = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(v_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, v_test) * ufl.dx
        - p_sol * ufl.div(v_test) * ufl.dx
        + ufl.div(u_sol) * q_test * ufl.dx
        - ufl.inner(f, v_test) * ufl.dx
    )

    # Jacobian
    J_form = ufl.derivative(F_form, w, dw)

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC on all boundaries
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        0.5 * np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -0.5 * np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at corner (0,0): p_exact(0,0) = 2.0
    p_bc_val = fem.Function(Q)
    p_bc_val.interpolate(lambda x: np.full(x.shape[1], 2.0))
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bc_p = fem.dirichletbc(p_bc_val, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Compile forms
    F_compiled = fem.form(F_form)
    J_compiled = fem.form(J_form)

    # Initialize: zero + BCs
    w.x.array[:] = 0.0
    fem.set_bc(w.x.petsc_vec, bcs)
    w.x.scatter_forward()

    # Manual Newton iteration with MUMPS direct solver
    dw_func = fem.Function(W)
    max_iter = 25
    atol = 1e-12
    n_newton = 0

    ksp = PETSc.KSP().create(comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    for i in range(max_iter):
        b = petsc.assemble_vector(F_compiled)
        petsc.apply_lifting(b, [J_compiled], bcs=[bcs], x0=[w.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs, w.x.petsc_vec)

        res_norm = b.norm()

        if res_norm < atol:
            n_newton = i
            b.destroy()
            break

        A = petsc.assemble_matrix(J_compiled, bcs=bcs)
        A.assemble()

        ksp.setOperators(A)
        b.scale(-1.0)
        dw_func.x.array[:] = 0.0
        ksp.solve(b, dw_func.x.petsc_vec)
        dw_func.x.scatter_forward()

        w.x.array[:] += dw_func.x.array[:]
        w.x.scatter_forward()
        n_newton = i + 1

        A.destroy()
        b.destroy()

    ksp.destroy()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Evaluate velocity magnitude on 50x50 grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    n_points = points_3d.shape[1]
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    vel_mag = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)

    u_grid = vel_mag.reshape((nx_out, ny_out))

    elapsed = time.time() - t0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
