import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    epsilon = 0.01
    beta_vec = [-12.0, 6.0]
    
    mesh_res = 120
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    u_ex = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    beta = ufl.as_vector(beta_vec)
    f = -epsilon * ufl.div(ufl.grad(u_ex)) + ufl.dot(beta, ufl.grad(u_ex))
    
    F_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
               + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
               - ufl.inner(f, v) * ufl.dx
               
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.dot(beta, beta))
    
    tau = h / (2.0 * vnorm)
    
    R = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    F_supg = F_galerkin + ufl.inner(R, tau * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a = ufl.lhs(F_supg)
    L = ufl.rhs(F_supg)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-9},
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
            
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": problem.solver.getIterationNumber()
    }
    
    return {"u": u_grid, "solver_info": solver_info}
