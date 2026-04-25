import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps = 0.05
    beta_vec = np.array([2.0, 1.0])
    t0 = 0.0
    t_end = 0.1
    dt = 0.001  # finer than suggested for accuracy
    n_steps = int(round((t_end - t0) / dt))

    # Mesh
    N = 160
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    # BCs
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # Functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0  # initial condition

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    beta = fem.Constant(domain, PETSc.ScalarType((beta_vec[0], beta_vec[1])))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    # Transient SUPG tau
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # Use tau including time term
    tau = ((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (4.0 * eps_c / h**2) ** 2) ** (-0.5)

    # Backward Euler weak form:
    # (u - u_n)/dt + beta·∇u - eps ∇²u = f
    # Galerkin:
    F_galerkin_a = (u * v / dt_c
                    + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v))
                    + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    F_galerkin_L = (u_n * v / dt_c + f_expr * v) * ufl.dx

    # SUPG additional terms: residual on LHS and RHS
    # residual = (u - u_n)/dt + beta·∇u - eps*div(grad u) - f
    # weighted by tau * (beta·∇v)
    # For P1, div(grad u) = 0 inside element
    R_lhs = u / dt_c + ufl.dot(beta, ufl.grad(u))
    R_rhs = u_n / dt_c + f_expr
    supg_weight = tau * ufl.dot(beta, ufl.grad(v))

    a_form = F_galerkin_a + R_lhs * supg_weight * ufl.dx
    L_form = F_galerkin_L + R_rhs * supg_weight * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_compiled.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)

    u_sol = fem.Function(V)
    total_iters = 0

    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    out = case_spec["output"]["grid"]
    nx = out["nx"]; ny = out["ny"]
    bbox = out["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny, nx)
    u_initial = np.zeros((ny, nx))

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    res = solve(case)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {res['u'].shape}")
    print(f"u range: [{res['u'].min():.4e}, {res['u'].max():.4e}]")
    print(f"Info: {res['solver_info']}")
