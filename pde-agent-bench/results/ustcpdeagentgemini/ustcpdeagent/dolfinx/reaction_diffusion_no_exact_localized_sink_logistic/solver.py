import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # 1. Parse parameters
    nx_mesh = case_spec.get("mesh_resolution", 128)
    ny_mesh = case_spec.get("mesh_resolution", 128)
    degree = case_spec.get("element_degree", 1)
    
    dt = case_spec.get("dt", 0.01)
    t_end = case_spec.get("t_end", 0.35)
    t0 = case_spec.get("t0", 0.0)
    
    out_nx = case_spec["output"]["grid"]["nx"]
    out_ny = case_spec["output"]["grid"]["ny"]
    out_bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    
    # 2. Setup Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Define Expressions and Initial Conditions
    x = ufl.SpatialCoordinate(domain)
    
    f = 4 * ufl.exp(-200 * ((x[0]-0.4)**2 + (x[1]-0.6)**2)) \
        - 2 * ufl.exp(-200 * ((x[0]-0.65)**2 + (x[1]-0.35)**2))
        
    u_n = fem.Function(V)
    u_initial = fem.Function(V)
    u0_expr = 0.4 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u0_expr_eval = fem.Expression(u0_expr, V.element.interpolation_points)
    u_n.interpolate(u0_expr_eval)
    u_initial.x.array[:] = u_n.x.array[:]
    
    # 4. Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_coord: np.full(x_coord.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 5. Variational Form
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    epsilon = case_spec.get("epsilon", 1.0)
    rho = case_spec.get("reaction_rho", 5.0)
    
    R_u = -rho * u * (1 - u)
    
    F = ufl.inner((u - u_n) / dt, v) * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(R_u, v) * ufl.dx \
        - ufl.inner(f, v) * ufl.dx
        
    J = ufl.derivative(F, u)
    
    # 6. Setup Nonlinear Solver
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J,
                                      petsc_options_prefix="rd_",
                                      petsc_options=petsc_options)
    
    # 7. Time Stepping Loop
    t = t0
    n_steps = 0
    nonlinear_iters = []
    
    while t < t_end - 1e-8:
        t += dt
        n_steps += 1
        
        problem.solve()
        
        try:
            num_its = problem.snes.getIterationNumber()
        except:
            num_its = 1
            
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        
        nonlinear_iters.append(num_its)
        
    # 8. Sample on uniform grid
    def sample_on_grid(u_func, nx, ny, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

        points_on_proc = []
        cells = []
        eval_map = []
        for i, pt in enumerate(pts):
            links = colliding.links(i)
            if len(links) > 0:
                points_on_proc.append(pt)
                cells.append(links[0])
                eval_map.append(i)

        u_vals = np.full(nx * ny, np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
            u_vals[eval_map] = vals.flatten()
            
        return u_vals.reshape((ny, nx))

    u_grid = sample_on_grid(u, out_nx, out_ny, out_bbox)
    u_init_grid = sample_on_grid(u_initial, out_nx, out_ny, out_bbox)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case = {
        "mesh_resolution": 64,
        "dt": 0.01,
        "t_end": 0.35,
        "output": {
            "grid": {
                "nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case)
    print(res["solver_info"])
