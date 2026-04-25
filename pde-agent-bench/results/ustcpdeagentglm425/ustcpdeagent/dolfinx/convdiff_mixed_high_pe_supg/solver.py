import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import os
os.environ["OMP_NUM_THREADS"] = "4"

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    eps = pde["params"]["epsilon"]
    beta = pde["params"]["beta"]
    
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    mesh_res = 320
    elem_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    laplacian_u = -2.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    beta_dot_grad_u = beta[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) + \
                      beta[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    
    f_source = -eps * laplacian_u + beta_dot_grad_u
    
    g_val = 0.0
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    h = ufl.CellDiameter(domain)
    beta_norm = np.sqrt(beta[0]**2 + beta[1]**2)
    delta_supg = h / (2.0 * beta_norm)
    
    beta_vec = ufl.as_vector([beta[0], beta[1]])
    
    a_galerkin = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
                 ufl.inner(beta_vec, ufl.grad(u)) * v * ufl.dx
    L_galerkin = f_source * v * ufl.dx
    
    a_supg = delta_supg * ufl.inner(beta_vec, ufl.grad(u)) * ufl.inner(beta_vec, ufl.grad(v)) * ufl.dx
    L_supg = delta_supg * f_source * ufl.inner(beta_vec, ufl.grad(v)) * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(g_val), boundary_dofs, V)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 3000,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    rtol_val = 1e-10
    
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    
    error_L2 = domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)),
        op=MPI.SUM
    )
    error_L2 = np.sqrt(error_L2)
    
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((3, nx_out * ny_out))
    pts[0, :] = XX.ravel()
    pts[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": rtol_val,
            "iterations": int(iterations),
        }
    }
