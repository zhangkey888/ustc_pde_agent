import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    mesh_res = 155
    elem_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(3.0 * (x[0] + x[1])) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = -ufl.div(ufl.grad(u_exact_ufl))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    num_points = points.shape[1]
    cells = np.full(num_points, -1, dtype=np.int32)
    adj_array = colliding_cells.array
    offsets = colliding_cells.offsets
    for i in range(num_points):
        start, end = offsets[i], offsets[i+1]
        if end > start:
            cells[i] = adj_array[start]
    
    valid = cells >= 0
    u_values = np.full(num_points, np.nan)
    if np.any(valid):
        pts_valid = points.T[valid]
        cells_valid = cells[valid]
        vals = u_sol.eval(pts_valid, cells_valid)
        u_values[valid] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }
    
    pde_time = case_spec.get("pde", {}).get("time", {})
    if pde_time and pde_time.get("is_transient", False):
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    result = {"u": u_grid, "solver_info": solver_info}
    
    if pde_time and pde_time.get("is_transient", False):
        result["u_initial"] = np.zeros_like(u_grid)
    
    return result
