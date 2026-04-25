import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse output grid specs
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Parse PDE parameters
    pde = case_spec["pde"]
    kappa_val = 1.0  # κ = 1.0
    
    # Mesh resolution and element degree
    # Use higher resolution to fully utilize time budget for max accuracy
    mesh_res = 140
    element_degree = 3

    # Create mesh with quadrilateral cells
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral,
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term using UFL for exact integration
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    f_ufl = (ufl.sin(6*pi*x[0])*ufl.sin(5*pi*x[1]) 
             + 0.4*ufl.sin(11*pi*x[0])*ufl.sin(9*pi*x[1]))
    
    # Bilinear and linear forms: -div(kappa * grad(u)) = f
    a = kappa_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx
    
    # Boundary conditions - homogeneous Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    # Solve using LinearProblem
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver iteration info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution onto the output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Build points array (3, N)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    # Gather across processes
    u_grid_global = np.zeros(nx_out * ny_out)
    comm.Allreduce(u_values, u_grid_global, op=MPI.SUM)
    u_grid = u_grid_global.reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    # Check if time-dependent
    if "time" in pde and pde.get("time", None) is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
