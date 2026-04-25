import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 40
    elem_degree = 2
    
    comm = MPI.COMM_WORLD
    p0 = np.array([bbox[0], bbox[2]], dtype=np.float64)
    p1 = np.array([bbox[1], bbox[3]], dtype=np.float64)
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res], 
                                    cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    kappa_ufl = 1.0 + 0.5 * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])
    u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    f_ufl = -ufl.div(kappa_ufl * ufl.grad(u_exact_ufl))
    
    kappa = fem.Function(V)
    kappa.interpolate(fem.Expression(kappa_ufl, V.element.interpolation_points))
    
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, 
        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            u_values[i] = u_sol.eval(points[:, i].reshape(1, 3), links[0])[0]
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)), op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")
        print(f"Iterations: {iterations}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "poisson", "kappa": {"type": "expr", "expr": "1 + 0.5*cos(2*pi*x)*cos(2*pi*y)"}, "time": None},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
