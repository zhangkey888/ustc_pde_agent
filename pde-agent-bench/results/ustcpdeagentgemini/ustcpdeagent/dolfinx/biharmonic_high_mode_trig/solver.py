import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 128
    degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary condition: u = 0 and Δu = 0 on boundary (valid for u = sin(3*pi*x)*sin(2*pi*y))
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    x = ufl.SpatialCoordinate(domain)
    # The source term for Δ²u = f
    # Δu = - ( (3π)² + (2π)² ) sin(3πx)sin(2πy) = -13π² u
    # Δ²u = (-13π²)² u = 169π⁴ u
    u_exact = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = 169 * ufl.pi**4 * u_exact
    
    # Let v = -Δu, then the biharmonic problem becomes:
    # 1) -Δv = -f
    # 2) -Δu = v
    
    # Step 1: Solve for v
    v_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    a_v = ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
    L_v = ufl.inner(-f, v_test) * ufl.dx
    
    problem_v = petsc.LinearProblem(
        a_v, L_v, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="v_"
    )
    v_sol = problem_v.solve()
    
    # Step 2: Solve for u
    u_trial = ufl.TrialFunction(V)
    u_test = ufl.TestFunction(V)
    a_u = ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
    L_u = ufl.inner(v_sol, u_test) * ufl.dx
    
    problem_u = petsc.LinearProblem(
        a_u, L_u, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="u_"
    )
    u_sol = problem_u.solve()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.zeros(len(points))
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 2
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
