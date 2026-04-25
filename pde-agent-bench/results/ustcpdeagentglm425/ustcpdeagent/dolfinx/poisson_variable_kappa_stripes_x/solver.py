import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse output grid specification
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]

    # Mesh resolution and element degree (adaptive: use budget for accuracy)
    mesh_res = 48
    elem_degree = 3

    # Create mesh on unit square
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Variable diffusion coefficient κ = 1 + 0.5*sin(6πx)
    kappa_func = fem.Function(V)
    kappa_func.interpolate(lambda x: 1.0 + 0.5 * np.sin(6 * np.pi * x[0]))

    # Source term f (analytically derived from manufactured solution u=sin(2πx)sin(πy))
    # f = 5π²(1+0.5sin(6πx))sin(2πx)sin(πy) - 6π²cos(6πx)cos(2πx)sin(πy)
    f_func = fem.Function(V)
    f_func.interpolate(lambda x: (
        5 * np.pi**2 * (1.0 + 0.5 * np.sin(6 * np.pi * x[0])) * np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
        - 6 * np.pi**2 * np.cos(6 * np.pi * x[0]) * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ))

    # Boundary condition g = sin(2πx)sin(πy) on ∂Ω
    g_func = fem.Function(V)
    g_func.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Dirichlet BC on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    # Variational form: ∫ κ∇u·∇v dx = ∫ fv dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_func * v * ufl.dx

    # Solve with direct LU
    rtol = 1e-12
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    # Vectorized extraction using AdjacencyList offsets/array
    N = pts.shape[1]
    offsets = colliding_cells.offsets
    arr = colliding_cells.array

    # Find valid points (at least one colliding cell)
    has_collision = offsets[1:] - offsets[:-1] > 0
    valid_indices = np.where(has_collision)[0]

    # Get first colliding cell for each valid point
    valid_cells = arr[offsets[valid_indices]]  # first link for each valid point
    valid_pts = pts.T[valid_indices]

    u_values = np.full(N, np.nan)
    if len(valid_pts) > 0:
        vals = u_sol.eval(valid_pts, valid_cells.astype(np.int32))
        u_values[valid_indices] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)), op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, mesh_res: {mesh_res}, degree: {elem_degree}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": 1,
        }
    }
