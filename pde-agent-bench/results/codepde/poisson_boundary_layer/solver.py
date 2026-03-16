import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("oracle_config", {}).get("pde", {})
    kappa_val = 1.0
    for coeff in pde_config.get("coefficients", []):
        if coeff.get("name") == "kappa" or coeff.get("name") == "κ":
            kappa_val = float(coeff["value"])

    # 2. Create mesh - use higher resolution for boundary layer (exp(6*x) grows fast)
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)

    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = exp(6*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(6 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = -kappa * laplacian(u_exact)
    # laplacian of exp(6x)*sin(pi*y) = 36*exp(6x)*sin(pi*y) - pi^2*exp(6x)*sin(pi*y)
    #                                 = (36 - pi^2)*exp(6x)*sin(pi*y)
    # So f = -kappa * (36 - pi^2)*exp(6x)*sin(pi*y)
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))
    f = -kappa * ufl.div(ufl.grad(u_exact_ufl))
    # Since -div(kappa*grad(u)) = f, we have f = -kappa * laplacian(u_exact)
    # But the equation is -div(kappa*grad(u)) = f, so the source is:
    f_source = -kappa * (36.0 - ufl.pi**2) * ufl.exp(6 * x[0]) * ufl.sin(ufl.pi * x[1])

    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_source * v * ufl.dx

    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(6 * x[0]) * np.sin(np.pi * x[1]))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # 7. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()

    # 8. Extract on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,
        }
    }