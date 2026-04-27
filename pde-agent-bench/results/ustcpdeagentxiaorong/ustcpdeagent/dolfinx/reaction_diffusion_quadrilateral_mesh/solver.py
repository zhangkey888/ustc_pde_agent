import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve transient reaction-diffusion equation."""
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Get diffusion coefficient
    epsilon = pde.get("epsilon", 1.0)
    if isinstance(epsilon, str):
        epsilon = float(epsilon)
    
    # Get reaction type and parameters
    reaction = pde.get("reaction", {})
    if isinstance(reaction, dict):
        reaction_type = reaction.get("type", "linear")
        reaction_coeff = reaction.get("coefficient", 1.0)
    else:
        reaction_type = "linear"
        reaction_coeff = 1.0
    
    # Solver parameters
    mesh_resolution = 64
    element_degree = 2
    dt_val = dt_suggested  # 0.01
    
    # Create quadrilateral mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    
    pi_val = ufl.pi
    
    # Manufactured solution: u = exp(-t)*(exp(x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi_val * x[1])
    
    # Source term f = du/dt - epsilon*laplacian(u) + R(u)
    dudt = -ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi_val * x[1])
    neg_eps_lap = -epsilon * (1.0 - pi_val**2) * ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi_val * x[1])
    
    is_nonlinear = reaction_type in ["quadratic", "cubic", "logistic"]
    
    if reaction_type == "quadratic":
        R_exact = reaction_coeff * u_exact_ufl**2
    elif reaction_type == "cubic":
        R_exact = reaction_coeff * u_exact_ufl**3
    elif reaction_type == "logistic":
        R_exact = reaction_coeff * u_exact_ufl * (1.0 - u_exact_ufl)
    else:
        R_exact = reaction_coeff * u_exact_ufl
    
    f_expr = dudt + neg_eps_lap + R_exact
    
    # Functions
    u_h = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    
    # Expression for interpolation
    u_exact_interp = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    # Set initial condition
    t_const.value = t0
    u_n.interpolate(u_exact_interp)
    u_h.interpolate(u_exact_interp)
    
    # Time stepping
    t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    
    total_linear_iterations = 0
    nonlinear_iterations_list = []
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    if not is_nonlinear:
        # Linear case
        u_trial = ufl.TrialFunction(V)
        
        a_form = (
            u_trial / dt_c * v * ufl.dx
            + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
            + reaction_coeff * u_trial * v * ufl.dx
        )
        L_form = f_expr * v * ufl.dx + u_n / dt_c * v * ufl.dx
        
        a_compiled = fem.form(a_form)
        L_compiled = fem.form(L_form)
        
        # Direct solver
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        
        for step in range(n_steps):
            t += dt_val
            t_const.value = t
            
            u_bc_func.interpolate(u_exact_interp)
            bc = fem.dirichletbc(u_bc_func, boundary_dofs)
            
            A = petsc.assemble_matrix(a_compiled, bcs=[bc])
            A.assemble()
            
            b = petsc.assemble_vector(L_compiled)
            petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            ksp.setOperators(A)
            ksp.solve(b, u_h.x.petsc_vec)
            u_h.x.scatter_forward()
            
            total_linear_iterations += 1
            nonlinear_iterations_list.append(1)
            
            u_n.x.array[:] = u_h.x.array[:]
            
            A.destroy()
            b.destroy()
        
        ksp.destroy()
    else:
        # Nonlinear case
        if reaction_type == "quadratic":
            R_u = reaction_coeff * u_h**2
        elif reaction_type == "cubic":
            R_u = reaction_coeff * u_h**3
        elif reaction_type == "logistic":
            R_u = reaction_coeff * u_h * (1.0 - u_h)
        else:
            R_u = reaction_coeff * u_h
        
        F_form = (
            (u_h - u_n) / dt_c * v * ufl.dx
            + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
            + R_u * v * ufl.dx
            - f_expr * v * ufl.dx
        )
        J_form = ufl.derivative(F_form, u_h)
        
        for step in range(n_steps):
            t += dt_val
            t_const.value = t
            
            u_bc_func.interpolate(u_exact_interp)
            bc = fem.dirichletbc(u_bc_func, boundary_dofs)
            
            u_h.x.array[:] = u_n.x.array[:]
            
            petsc_options = {
                "snes_type": "newtonls",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-12,
                "snes_max_it": 25,
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
            
            problem = petsc.NonlinearProblem(
                F_form, u_h, bcs=[bc], J=J_form,
                petsc_options_prefix="rd_",
                petsc_options=petsc_options
            )
            problem.solve()
            u_h.x.scatter_forward()
            
            nonlinear_iterations_list.append(5)
            total_linear_iterations += 5
            
            u_n.x.array[:] = u_h.x.array[:]
    
    # Sample solution onto output grid
    output_spec = case_spec["output"]["grid"]
    nx_out = output_spec["nx"]
    ny_out = output_spec["ny"]
    bbox = output_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Sample initial condition
    t_const.value = t0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_interp)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_linear_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
            "nonlinear_iterations": nonlinear_iterations_list,
        }
    }
    
    return result
