import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

def solve(case_spec: dict) -> dict:
    # Setup
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = case_spec.get("output", {}).get("grid", {}).get("nx", 64)
    ny = case_spec.get("output", {}).get("grid", {}).get("ny", 64)
    bbox = case_spec.get("output", {}).get("grid", {}).get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    t0 = 0.0
    t_end = 0.1
    dt = 0.005 # Refined step for better accuracy
    
    mesh_res = 100 # Refined spatial resolution
    elem_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_h = fem.Function(V)
    u_h.name = "u"
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for initial, BC, and source
    t_sym = fem.Constant(domain, PETSc.ScalarType(t0))
    u_exact = ufl.exp(-t_sym) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # kappa
    kappa = 1.0 + 0.3 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # Source term f = u_t - div(kappa * grad(u))
    f = -ufl.exp(-t_sym) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]) \
        - ufl.div(kappa * ufl.grad(u_exact))
        
    # Initial condition
    expr_u0 = fem.Expression(u_exact, V.element.interpolation_points)
    u_n.interpolate(expr_u0)
    
    # Save initial state
    def sample_grid(func):
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
        for i in range(len(pts)):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(pts[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
                
        u_values = np.full((len(pts),), np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        u_global = np.zeros_like(u_values)
        comm.Reduce(np.nan_to_num(u_values, nan=0.0), u_global, op=MPI.SUM, root=0)
        mask = np.zeros_like(u_values)
        comm.Reduce(np.where(np.isnan(u_values), 0.0, 1.0), mask, op=MPI.SUM, root=0)
        
        if comm.rank == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                u_global = u_global / mask
            return u_global.reshape((ny, nx))
        return None

    u_init_grid = sample_grid(u_n)
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(expr_u0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = ufl.inner(u, v) * ufl.dx + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_c * ufl.inner(f, v) * ufl.dx
    
    # Setup solver
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-8)
    
    t = t0
    n_steps = int(np.round((t_end - t0) / dt))
    
    total_iters = 0
    for i in range(n_steps):
        t += dt
        t_sym.value = t
        
        # update BC
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array
        
    u_final_grid = sample_grid(u_h)
    
    if comm.rank == 0:
        return {
            "u": u_final_grid,
            "u_initial": u_init_grid,
            "solver_info": {
                "mesh_resolution": mesh_res,
                "element_degree": elem_degree,
                "ksp_type": "cg",
                "pc_type": "hypre",
                "rtol": 1e-8,
                "iterations": total_iters,
                "dt": dt,
                "n_steps": n_steps,
                "time_scheme": "backward_euler"
            }
        }
    return {}

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    if res and "u" in res:
        print("Max u:", np.max(res["u"]))
