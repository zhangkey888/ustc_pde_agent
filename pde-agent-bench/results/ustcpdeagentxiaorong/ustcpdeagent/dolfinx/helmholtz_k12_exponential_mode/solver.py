import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    k_val = case_spec["pde"]["parameters"]["k"]
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    elem_degree = 3
    mesh_res = 40
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                   cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    u_exact = ufl.exp(x[0]) * ufl.cos(2 * pi_val * x[1])
    
    k_const = fem.Constant(domain, ScalarType(k_val))
    
    f_expr = (4.0 * pi_val**2 - 1.0 - k_const**2) * u_exact
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
