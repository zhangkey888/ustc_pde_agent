import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    epsilon = pde_config.get("epsilon", 0.3)
    beta = pde_config.get("beta", [0.5, 0.3])
    
    nx_out = case_spec.get("nx", 50)
    ny_out = case_spec.get("ny", 50)
    
    # 2. Create mesh
    mesh_res = 64
    elem_degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u_exact = x*(1-x)*y*(1-y)
    u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    
    # Compute source term f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    # grad(u_exact) = ((1-2x)*y*(1-y), x*(1-x)*(1-2y))
    # laplacian(u_exact) = -2*y*(1-y) - 2*x*(1-x)
    # So -epsilon * laplacian = epsilon * (2*y*(1-y) + 2*x*(1-x))
    # beta . grad = beta[0]*(1-2x)*y*(1-y) + beta[1]*x*(1-x)*(1-2y)
    
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    f = -eps_const * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # Bilinear form: epsilon * inner(grad(u), grad(v)) + (beta . grad(u)) * v
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # 5. Boundary conditions (u = 0 on boundary since exact solution vanishes there)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="cdiff_"
    )
    uh = problem.solve()
    
    # 7. Extract solution on uniform grid
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
    
    u_values = np.zeros(points_3d.shape[0])
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
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,
        }
    }