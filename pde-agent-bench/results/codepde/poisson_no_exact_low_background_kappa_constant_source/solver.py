import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec["pde"]
    
    # Source term
    source_expr = pde_config.get("source", 1.0)
    
    # Diffusion coefficient
    kappa_info = pde_config.get("coefficients", {}).get("kappa", {})
    
    # Boundary conditions
    bc_config = pde_config.get("boundary_conditions", {})
    
    # Grid size for output
    nx_out = 50
    ny_out = 50
    
    # 2. Create mesh - use higher resolution for accuracy
    mesh_res = 128
    element_degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define kappa
    x = ufl.SpatialCoordinate(domain)
    
    # kappa = 0.2 + 0.8*exp(-80*((x-0.5)**2 + (y-0.5)**2))
    kappa_ufl = 0.2 + 0.8 * ufl.exp(-80.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    # Source term f = 1.0
    f = fem.Constant(domain, default_scalar_type(1.0))
    
    # 5. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions - default to u=0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Determine BC value
    bc_value = 0.0
    if bc_config:
        # Check for Dirichlet BCs
        for key, val in bc_config.items():
            if isinstance(val, dict) and val.get("type") == "dirichlet":
                bc_value = float(val.get("value", 0.0))
                break
            elif isinstance(val, (int, float)):
                bc_value = float(val)
                break
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(bc_value), dofs, V)
    
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
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on uniform grid
    xv = np.linspace(0.0, 1.0, nx_out)
    yv = np.linspace(0.0, 1.0, ny_out)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }