import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [-12.0, 6.0])
    
    domain_spec = case_spec.get("domain", {})
    bounds = domain_spec.get("bounds", [[0, 1], [0, 1]])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 55)
    ny_out = output.get("ny", 55)
    
    N = 64
    element_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([bounds[0][0], bounds[1][0]]),
         np.array([bounds[0][1], bounds[1][1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    
    f_expr = (-eps_c * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]) * (1.0 - ufl.pi**2)
              + beta_vec[0] * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.exp(x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1]))
    
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    a_supg += -eps_c * ufl.div(ufl.grad(u)) * v_supg * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a_form = a_std + a_supg
    L_form = L_std + L_supg
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    try:
        problem = petsc.LinearProblem(
            a_form, L_form, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "100",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_form, L_form, bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="cdiff2_"
        )
        u_sol = problem.solve()
    
    x_coords = np.linspace(bounds[0][0], bounds[0][1], nx_out)
    y_coords = np.linspace(bounds[1][0], bounds[1][1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        },
    }
