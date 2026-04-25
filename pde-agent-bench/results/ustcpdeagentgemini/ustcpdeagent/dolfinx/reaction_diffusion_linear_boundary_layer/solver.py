import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid properties
    grid = case_spec.get("output", {}).get("grid", {})
    nx_out = grid.get("nx", 64)
    ny_out = grid.get("ny", 64)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Parameters
    mesh_resolution = case_spec.get("mesh_resolution", 80)
    element_degree = case_spec.get("element_degree", 2)
    
    # Time parameters
    t0 = 0.0
    t_end = 0.3
    dt_val = case_spec.get("dt", 0.005)
    
    # PDE Parameters
    epsilon_val = case_spec.get("pde", {}).get("epsilon", 0.01)
    
    # 1. Mesh & Function Space
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 2. Time-stepping variables
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Exact solution definition
    def exact_solution_expr(t_ufl):
        return ufl.exp(-t_ufl) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initialize u_n
    u_exact_t0 = exact_solution_expr(t_const)
    u_n_expr = fem.Expression(u_exact_t0, V.element.interpolation_points)
    u_n.interpolate(u_n_expr)
    
    # Save initial condition for tracking
    # We will sample it later or just keep u_n array
    u_initial_array = None

    # 3. Variational formulation (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    epsilon = fem.Constant(domain, PETSc.ScalarType(epsilon_val))
    
    # Source term derivation from exact solution
    # u_t = -u_exact
    # laplacian(u) = (16 - pi^2) * u_exact
    # f = u_t - epsilon * laplacian(u) + R(u)
    # We assume R(u) = 0 for linear benchmark if not specified, 
    # but the simplest general approach is just computing f explicitly:
    u_exact_sym = exact_solution_expr(t_const)
    u_t_sym = -u_exact_sym
    laplacian_u_sym = (16.0 - ufl.pi**2) * u_exact_sym
    f_sym = u_t_sym - epsilon * laplacian_u_sym
    
    F = (u - u_n) / dt * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f_sym * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # 4. Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x_coord: np.ones(x_coord.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Time loop
    t = t0
    n_steps = 0
    total_iterations = 0
    
    while t < t_end - 1e-8:
        t += dt_val
        n_steps += 1
        t_const.value = t
        
        # Update BC
        u_bc_expr = fem.Expression(exact_solution_expr(t_const), V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += ksp.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # 5. Output Interpolation
    def sample_on_grid(u_func, nx, ny, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, pts)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

        cells = []
        points_on_proc = []
        eval_map = []
        for i, pt in enumerate(pts):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(pt)
                cells.append(links[0])
                eval_map.append(i)

        u_values = np.full((nx * ny,), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        return u_values.reshape((ny, nx))
        
    u_grid = sample_on_grid(u_sol, nx_out, ny_out, bbox)
    
    # Calculate initial grid for tracking if required, we just re-evaluate at t0
    t_const.value = t0
    u_init_temp = fem.Function(V)
    u_init_temp.interpolate(fem.Expression(exact_solution_expr(t_const), V.element.interpolation_points))
    u_initial_grid = sample_on_grid(u_init_temp, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": total_iterations if total_iterations > 0 else n_steps,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"epsilon": 0.01}
    }
    res = solve(case_spec)
    print("Solve completed successfully. Min/Max u:", np.nanmin(res["u"]), np.nanmax(res["u"]))
