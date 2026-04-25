import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid info
    grid = case_spec.get("output", {}).get("grid", {})
    nx = grid.get("nx", 50)
    ny = grid.get("ny", 50)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Solver parameters
    mesh_res = 120
    degree = 2
    dt = 0.005
    t_end = 0.3
    epsilon = 1.0
    reaction_coeff = 1.0
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[bbox[0], bbox[2]], [bbox[1], bbox[3]]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Exact solution for manufactured solution
    u_ex = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    # Set initial condition
    u_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_n.interpolate(u_expr)
    
    # Sample initial condition for tracking
    def get_eval(func):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]
        
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
        MPI.COMM_WORLD.Reduce(u_values, u_global, op=MPI.MAX, root=0) # using MAX assuming disjoint support where non-nan
        
        return u_global.reshape(ny, nx)

    u_initial = get_eval(u_n).copy()
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_expr)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Crank-Nicolson parameters
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    theta = 0.5
    
    # PDE: u_t - epsilon * nabla^2 u + reaction_coeff * u = f
    # F = u_t - eps * div(grad(u)) + r * u - f = 0
    # For manufactured f, we derive f analytically from u_ex
    f_ex = -u_ex - epsilon * ufl.div(ufl.grad(u_ex)) + reaction_coeff * u_ex
    
    # Form: Crank-Nicolson
    # (u - u_n)/dt - eps * div(grad(u_mid)) + r * u_mid = f_mid
    u_mid = theta * u + (1 - theta) * u_n
    t_mid = t + theta * dt
    
    f_mid_ex = ufl.replace(f_ex, {t: t_mid})
    
    F = (u - u_n) / dt_c * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx \
        + reaction_coeff * u_mid * v * ufl.dx \
        - f_mid_ex * v * ufl.dx
        
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    u_sol = fem.Function(V)
    
    t_current = 0.0
    n_steps = int(np.round(t_end / dt))
    
    iterations = 0
    
    for step in range(n_steps):
        t_current += dt
        t.value = t_current
        
        u_bc.interpolate(u_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        iterations += ksp.getIterationNumber() or 1
        
        u_n.x.array[:] = u_sol.x.array
        
    u_final = get_eval(u_sol)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson"
    }
    
    return {
        "u": u_final,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    print("Done. Output shape:", res["u"].shape)
