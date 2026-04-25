import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # 1. Mesh and Function Space
    mesh_resolution = 128
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # 2. Parameters
    epsilon = 0.01
    beta_vec = [10.0, 5.0]
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    
    # 3. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 4. Variational Problem (with SUPG)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Source term: f = exp(-200*((x-0.3)**2 + (y-0.7)**2))
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Standard Galerkin formulation
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # SUPG Stabilization
    h = ufl.CellDiameter(domain)
    v_norm = ufl.sqrt(ufl.inner(beta, beta))
    
    # Simple SUPG parameter for convection-dominated flow
    tau = h / (2.0 * v_norm)
    
    # Residual of the strong equation
    # Note: for P1 elements, Laplacian is zero, so -epsilon * div(grad(u)) is omitted
    R = ufl.inner(beta, ufl.grad(u)) - f_expr
    
    # SUPG test function perturbation
    v_supg = tau * ufl.inner(beta, ufl.grad(v))
    
    # Add stabilization terms
    a_stab = a + ufl.inner(ufl.inner(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_stab = L + ufl.inner(f_expr, v_supg) * ufl.dx
    
    # 5. Solve Linear Problem
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a_stab, L_stab, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Extract linear iterations (approximate since petsc.LinearProblem doesn't directly expose it easily)
    # Actually, we can get it from the KSP object inside the solver if needed. For now, set a dummy or try to access.
    iterations = problem.solver.getIterationNumber()
    
    # 6. Evaluate on output grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
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
            
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations
        }
    }
