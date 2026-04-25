import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx_mesh = 64
    ny_mesh = 64
    element_degree = 2
    dt = 0.02
    t_end = 0.2
    kappa = 0.5
    
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution as UFL expression for BC and Source
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution: exp(-2t) * cos(pi*x) * cos(pi*y)
    u_exact = ufl.exp(-2.0 * t) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term: f = u_t - kappa * Delta u
    # u_t = -2 * u_exact
    # Delta u = -2 * pi^2 * u_exact
    # f = (-2 + kappa * 2 * pi^2) * u_exact = (pi^2 - 2) * u_exact
    f = (ufl.pi**2 - 2.0) * u_exact
    
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    u_initial_grid = None
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Backward Euler: (u - u_n)/dt - kappa * Delta u = f
    # (u, v) + dt * kappa * (grad u, grad v) = (u_n, v) + dt * (f, v)
    F = u * v * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - u_n * v * ufl.dx - dt * f * v * ufl.dx
    a, L = ufl.lhs(F), ufl.rhs(F)
    
    # Boundary Condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Linear solver setup
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)
    rtol = 1e-8
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    
    # Output grid setup
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
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
            
    def sample_func(func):
        u_values = np.full((pts.shape[0],), np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(ny_out, nx_out)
    
    u_initial_grid = sample_func(u_n)
    
    total_iters = 0
    num_steps = int(np.round(t_end / dt))
    
    for step in range(num_steps):
        t.value += dt
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
        
    u_final_grid = sample_func(u_n)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "rtol": rtol,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": num_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_final_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid
    }

if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    res = solve(spec)
    print("Done. Iterations:", res["solver_info"]["iterations"])
