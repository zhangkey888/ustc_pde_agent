import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    # Grid info
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh config
    mesh_res = 128
    elem_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution for BC and RHS
    u_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Parameters
    epsilon = 0.05
    beta = ufl.as_vector([4.0, 2.0])
    
    # RHS f = -eps * div(grad(u_ex)) + beta . grad(u_ex)
    grad_u_ex = ufl.grad(u_ex)
    div_grad_u_ex = ufl.div(grad_u_ex)
    f = -epsilon * div_grad_u_ex + ufl.inner(beta, grad_u_ex)
    
    # Standard weak form
    F_std = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx \
          - f * v * ufl.dx
          
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    
    # tau = h / (2 * |beta|)
    tau = h / (2.0 * beta_norm)
    
    # Strong residual
    R = -epsilon * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u)) - f
    
    # Test function modification
    v_supg = tau * ufl.inner(beta, ufl.grad(v))
    
    # Full form
    F = F_std + R * v_supg * ufl.dx
    
    a, L = ufl.lhs(F), ufl.rhs(F)
    
    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    bdry_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    expr = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(expr)
    bc = fem.dirichletbc(u_bc, bdry_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="cd_"
    )
    
    u_sol = problem.solve()
    
    # Evaluate at grid points
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)].T
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_out = np.full(nx*ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
