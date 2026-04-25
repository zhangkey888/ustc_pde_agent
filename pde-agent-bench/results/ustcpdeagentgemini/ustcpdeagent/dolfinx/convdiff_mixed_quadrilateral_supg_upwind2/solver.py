import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # 1. Parse parameters
    eps_val = 0.005
    beta_val = [18.0, 6.0]
    
    output_grid = case_spec.get("output", {}).get("grid", {})
    nx_out = output_grid.get("nx", 50)
    ny_out = output_grid.get("ny", 50)
    bbox = output_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # We choose mesh resolution and element degree
    mesh_res = 120
    degree = 2
    
    # 2. Mesh and Function Space
    comm = MPI.COMM_WORLD
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Manufactured Solution and Source Term
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    eps = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(tuple(beta_val)))
    
    # -eps * div(grad(u_ex)) + beta . grad(u_ex)
    grad_u_ex = ufl.grad(u_ex)
    div_grad_u_ex = ufl.div(grad_u_ex)
    f_expr = -eps * div_grad_u_ex + ufl.dot(beta, grad_u_ex)
    
    # 4. Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 5. Variational Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Galerkin part
    a_galerkin = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_galerkin = ufl.inner(f_expr, v) * ufl.dx
    
    # SUPG Stabilization
    # tau = h / (2 * |beta|)
    # Approximating h
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm)
    
    residual = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(-eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = ufl.inner(f_expr, v_supg) * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # 6. Solve Linear Problem
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Retrieve number of iterations
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Interpolate onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
