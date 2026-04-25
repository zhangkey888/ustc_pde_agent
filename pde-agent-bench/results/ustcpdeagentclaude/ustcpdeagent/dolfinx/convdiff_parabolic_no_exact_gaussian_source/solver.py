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
    beta_vec = np.array([6.0, 2.0], dtype=np.float64)
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.001  # finer than suggested 0.02
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 200
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    # BC: u=0 on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    # Trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    # Time
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(eps))
    beta = fem.Constant(domain, PETSc.ScalarType((beta_vec[0], beta_vec[1])))

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)) * ufl.exp(-t_const)

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1e-14)
    # tau for transient CD with SUPG (standard formula)
    Pe_h = beta_norm * h / (2.0 * eps_const)
    # tau: classic form
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h + 1e-10) - 1.0 / (Pe_h + 1e-10))
    # To be safe, use simpler robust form:
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_h + 1e-10) - 1.0 / (Pe_h + 1e-10))

    # Galerkin residual-based: 
    # (u - u_n)/dt - eps*div(grad u) + beta.grad u - f = 0
    # Weak form (backward Euler) with SUPG:
    # a(u,v) + sum_K tau*(u/dt - eps*div(grad u) + beta.grad u) * (beta.grad v) dK
    # = L(v) + sum_K tau*(u_n/dt + f) * (beta.grad v) dK

    # Galerkin part
    F_gal_lhs = (u / dt_const) * v * ufl.dx \
                + eps_const * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
                + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    F_gal_rhs = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx

    # SUPG part (strong residual times tau * beta.grad(v))
    # Using P1: div(grad u) = 0, so strong residual simplifies
    R_strong_lhs = u / dt_const + ufl.dot(beta, ufl.grad(u)) - eps_const * ufl.div(ufl.grad(u))
    R_strong_rhs = u_n / dt_const + f_expr

    F_supg_lhs = tau * R_strong_lhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    F_supg_rhs = tau * R_strong_rhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a = F_gal_lhs + F_supg_lhs
    L = F_gal_rhs + F_supg_rhs

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble LHS once
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)

    u_sol = fem.Function(V)
    total_iters = 0

    for step in range(n_steps):
        t_new = t0 + (step + 1) * dt_val
        t_const.value = t_new

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
    out_grid = case_spec["output"]["grid"]
    nx = out_grid["nx"]
    ny = out_grid["ny"]
    bbox = out_grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

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

    u_grid = u_values.reshape(ny, nx)
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
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print("Wall time:", elapsed)
    print("u shape:", result["u"].shape)
    print("u min/max:", result["u"].min(), result["u"].max())
    print("solver_info:", result["solver_info"])
