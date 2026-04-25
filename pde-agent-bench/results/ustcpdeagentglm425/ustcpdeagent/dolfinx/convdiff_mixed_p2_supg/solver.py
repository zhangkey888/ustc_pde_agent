import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    eps = float(case_spec["pde"]["diffusion"])
    beta = [float(b) for b in case_spec["pde"]["velocity"]]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    N = 128
    degree = 3
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    beta_ufl = ufl.as_vector(beta)
    f_ufl = (5.0*eps*pi**2*ufl.sin(pi*x[0])*ufl.sin(2*pi*x[1]) 
             + 10.0*pi*ufl.cos(pi*x[0])*ufl.sin(2*pi*x[1]) 
             + 8.0*pi*ufl.sin(pi*x[0])*ufl.cos(2*pi*x[1]))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Standard Galerkin formulation (SUPG not needed for smooth manufactured solution)
    a = eps*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + ufl.dot(beta_ufl, ufl.grad(u))*v*ufl.dx
    L = f_ufl*v*ufl.dx
    u_exact_ufl = ufl.sin(pi*x[0])*ufl.sin(2*pi*x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    rtol = 1e-10
    ksp_type = "gmres"
    pc_type = "ilu"
    problem = petsc.LinearProblem(a, L, bcs=[bc], 
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)},
        petsc_options_prefix="cd_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    iterations = problem.solver.getIterationNumber()
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    u_values = np.full((nx_out*ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape(ny_out, nx_out)
    # Verification: compute L2 error
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    error_func = fem.Function(V)
    error_func.x.array[:] = u_sol.x.array - u_exact_func.x.array
    L2_sq = fem.assemble_scalar(fem.form(ufl.inner(error_func, error_func)*ufl.dx))
    L2_err = np.sqrt(comm.allreduce(L2_sq, op=MPI.SUM))
    # Grid error
    u_exact_grid = np.sin(np.pi*XX)*np.sin(2*np.pi*YY)
    max_err_grid = np.max(np.abs(u_grid - u_exact_grid))
    if comm.rank == 0:
        print(f"[ConvDiff] P{degree} N={N} eps={eps} L2={L2_err:.3e} MaxGrid={max_err_grid:.3e} Iters={iterations}")
    return {"u": u_grid, "solver_info": {"mesh_resolution": N, "element_degree": degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iterations}}
