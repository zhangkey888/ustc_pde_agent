import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Source term
    source_term = pde_config.get("source_term", "1.0")
    f_val = float(source_term) if isinstance(source_term, (int, float, str)) else 1.0
    
    # Diffusion coefficient
    coefficients = pde_config.get("coefficients", {})
    kappa_val = float(coefficients.get("kappa", 1.0))
    
    # Boundary conditions
    bcs_spec = pde_config.get("boundary_conditions", {})
    
    # Grid size for output
    nx_out = 50
    ny_out = 50
    
    # Mesh resolution and element degree
    mesh_resolution = 128
    element_degree = 2
    
    # 2. Create mesh (quadrilateral as specified in case ID)
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.quadrilateral
    )
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(f_val))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions - homogeneous Dirichlet on all boundaries (g=0 by default)
    # Parse boundary conditions from spec
    g_val = 0.0
    if bcs_spec:
        for key, bc_info in bcs_spec.items():
            if isinstance(bc_info, dict):
                bc_type = bc_info.get("type", "dirichlet")
                if bc_type.lower() == "dirichlet":
                    g_val = float(bc_info.get("value", 0.0))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(g_val), boundary_dofs, V)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    x_coords = np.linspace(0.0, 1.0, nx_out)
    y_coords = np.linspace(0.0, 1.0, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = xx.ravel()
    points[1, :] = yy.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }