import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- Extract parameters ----
    pde_params = case_spec.get("pde", {}).get("parameters", {})
    eps = float(pde_params.get("eps", pde_params.get("epsilon", pde_params.get("diffusivity", 0.03))))

    beta_raw = pde_params.get("beta", pde_params.get("velocity", [5.0, 2.0]))
    if isinstance(beta_raw, (list, tuple)):
        beta_vec = np.array(beta_raw, dtype=float)
    elif isinstance(beta_raw, np.ndarray):
        beta_vec = beta_raw.astype(float)
    else:
        beta_vec = np.array([5.0, 2.0], dtype=float)

    beta_ufl = ufl.as_vector(beta_vec.tolist())

    # Output grid
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]

    # ---- Solver parameters ----
    N = 160
    p_deg = 2
    rt = 1e-12

    # ---- Create mesh ----
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", p_deg))

    x = ufl.SpatialCoordinate(domain)

    # ---- Manufactured solution: u = sin(pi*x)*sin(2*pi*y) ----
    # laplacian(u) = -pi^2*sin(pi*x)*sin(2*pi*y) - 4*pi^2*sin(pi*x)*sin(2*pi*y) = -5*pi^2*sin(pi*x)*sin(2*pi*y)
    # beta.grad(u) = beta_x*pi*cos(pi*x)*sin(2*pi*y) + beta_y*2*pi*sin(pi*x)*cos(2*pi*y)
    # f = -eps*laplacian(u) + beta.grad(u)
    #   = 5*eps*pi^2*sin(pi*x)*sin(2*pi*y) + beta_x*pi*cos(pi*x)*sin(2*pi*y) + 2*beta_y*pi*sin(pi*x)*cos(2*pi*y)

    f_val = (5.0 * eps * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
             + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
             + 2.0 * beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]))

    # ---- Boundary condition: u = sin(pi*x)*sin(2*pi*y) = 0 on entire boundary ----
    fdim = domain.topology.dim - 1
    bnd_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bnd_dofs = fem.locate_dofs_topological(V, fdim, bnd_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bnd_dofs, V)

    # ---- Variational form with SUPG stabilization ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a_diff = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_conv = ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_K = beta_norm * h / (2.0 * eps)

    # SUPG parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_K + 1e-30) - 1.0 / (Pe_K + 1e-30))

    # Stabilized bilinear form
    a_supg = tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(v)),
                              ufl.dot(beta_ufl, ufl.grad(u))) * ufl.dx
    a = a_diff + a_conv + a_supg

    # Stabilized linear form
    L_std = ufl.inner(f_val, v) * ufl.dx
    L_supg = tau * ufl.inner(ufl.dot(beta_ufl, ufl.grad(v)), f_val) * ufl.dx
    L = L_std + L_supg

    # ---- Solve ----
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iters = problem.solver.getIterationNumber()

    # ---- Evaluate on output grid ----
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    n_pts = nx_out * ny_out
    points = np.zeros((n_pts, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

    domain.topology.create_connectivity(domain.topology.dim, 0)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    num_local_cells = domain.topology.index_map(domain.topology.dim).size_local

    cell_cands = geometry.compute_collisions_points(bb_tree, points)
    col_cells = geometry.compute_colliding_cells(domain, cell_cands, points)

    pts_on_proc = []
    cells_on_proc = []
    idx_on_proc = []
    for i in range(n_pts):
        links = col_cells.links(i)
        for c in links:
            if c < num_local_cells:
                pts_on_proc.append(points[i])
                cells_on_proc.append(c)
                idx_on_proc.append(i)
                break

    u_loc = np.zeros(n_pts)
    f_loc = np.zeros(n_pts)

    if len(pts_on_proc) > 0:
        pa = np.array(pts_on_proc)
        ca = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pa, ca).ravel()
        u_loc[np.array(idx_on_proc)] = vals
        f_loc[np.array(idx_on_proc)] = 1.0

    u_glob = np.zeros(n_pts)
    f_glob = np.zeros(n_pts)
    comm.Allreduce(u_loc, u_glob, op=MPI.SUM)
    comm.Allreduce(f_loc, f_glob, op=MPI.SUM)

    # Handle missing points (boundary corners etc.): nudge inward
    missing = np.where(f_glob == 0)[0]
    if len(missing) > 0:
        eps_nudge = 1e-8
        nudged = points[missing].copy()
        nudged[:, 0] = np.clip(nudged[:, 0], xmin + eps_nudge, xmax - eps_nudge)
        nudged[:, 1] = np.clip(nudged[:, 1], ymin + eps_nudge, ymax - eps_nudge)

        cell_cands2 = geometry.compute_collisions_points(bb_tree, nudged)
        col_cells2 = geometry.compute_colliding_cells(domain, cell_cands2, nudged)

        pts2 = []
        cells2 = []
        idx2 = []
        for j in range(len(missing)):
            links = col_cells2.links(j)
            for c in links:
                if c < num_local_cells:
                    pts2.append(nudged[j])
                    cells2.append(c)
                    idx2.append(j)
                    break

        u_loc2 = np.zeros(n_pts)
        f_loc2 = np.zeros(n_pts)
        if len(pts2) > 0:
            v2 = u_sol.eval(np.array(pts2), np.array(cells2, dtype=np.int32)).ravel()
            u_loc2[missing[np.array(idx2)]] = v2
            f_loc2[missing[np.array(idx2)]] = 1.0

        u_g2 = np.zeros(n_pts)
        f_g2 = np.zeros(n_pts)
        comm.Allreduce(u_loc2, u_g2, op=MPI.SUM)
        comm.Allreduce(f_loc2, f_g2, op=MPI.SUM)

        m2 = f_g2 > 0
        u_glob[m2] = u_g2[m2]
        f_glob[m2] = f_g2[m2]

    # Average where multiple procs found the point
    valid = f_glob > 0
    u_glob[valid] /= f_glob[valid]

    u_grid = u_glob.reshape(ny_out, nx_out)

    # ---- L2 error verification ----
    u_ex_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    u_ex_f = fem.Function(V)
    u_ex_f.interpolate(fem.Expression(u_ex_ufl, V.element.interpolation_points))

    err2 = fem.assemble_scalar(fem.form((u_sol - u_ex_f)**2 * ufl.dx))
    L2_err = np.sqrt(comm.allreduce(err2, op=MPI.SUM))

    if rank == 0:
        print(f"N={N}, P{p_deg}, eps={eps}, beta={beta_vec.tolist()}, L2 error = {L2_err:.6e}")
        u_exact_grid = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
        mx_err = np.max(np.abs(u_grid - u_exact_grid))
        print(f"Max grid error = {mx_err:.6e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": p_deg,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rt,
            "iterations": int(iters),
        }
    }
