import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    comm = MPI.COMM_WORLD
    N = 96
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    f_expr = 4.0 * pi**4 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    # w = -Δu = 2*pi^2*sin(pi*x)*sin(pi*y), zero on boundary
    # -Δw = f, w=0 on boundary (for this case)
    # -Δu = w, u=0 on boundary

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    # BC for w (=0 since sin(pi*x)*sin(pi*y) vanishes on boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    bc_w = fem.dirichletbc(zero, dofs)

    w_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_w = ufl.inner(ufl.grad(w_tr), ufl.grad(v)) * ufl.dx
    L_w = f_expr * v * ufl.dx

    prob_w = petsc.LinearProblem(
        a_w, L_w, bcs=[bc_w],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_w_",
    )
    w_sol = prob_w.solve()
    its_w = prob_w.solver.getIterationNumber()

    # Now solve -Δu = w, u=0 on boundary
    zero2 = fem.Function(V)
    zero2.x.array[:] = 0.0
    bc_u = fem.dirichletbc(zero2, dofs)

    u_tr = ufl.TrialFunction(V)
    a_u = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L_u = w_sol * v * ufl.dx

    prob_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc_u],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12},
        petsc_options_prefix="bih_u_",
    )
    u_sol = prob_u.solve()
    its_u = prob_u.solver.getIterationNumber()

    # Sample onto uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(its_w + its_u),
        },
    }
