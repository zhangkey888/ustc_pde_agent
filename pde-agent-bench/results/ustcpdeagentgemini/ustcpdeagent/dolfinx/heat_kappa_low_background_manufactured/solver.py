import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract case params
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    t0 = 0.0
    t_end = 0.1
    dt = 0.01
    
    # Mesh and function space
    res = 64
    degree = 2
    domain = mesh.create_unit_square(comm, res, res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution for initial, BC, and source
    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, PETSc.ScalarType(t0))
    u_ex = ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    kappa_expr = 0.2 + ufl.exp(-120.0 * ((x[0]-0.55)**2 + (x[1]-0.45)**2))
    f_expr = -u_ex - ufl.div(kappa_expr * ufl.grad(u_ex))
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    
    # Create u_initial for output
    u_initial = np.copy(u_n.x.array)
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = u * v * ufl.dx - u_n * v * ufl.dx + dt_c * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_c * f_expr * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Matrix and Vector
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-8)
    
    # Time stepping
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    t = t0
    n_steps = int(np.round((t_end - t0) / dt))
    total_iters = 0
    
    for i in range(n_steps):
        t += dt
        t_c.value = t
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
        
    # Interpolation on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
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
            
    u_out = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny_out, nx_out))
    
    # Same for initial condition
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u_initial
    u_init_out = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_init_out[eval_map] = vals_init.flatten()
    u_init_grid = u_init_out.reshape((ny_out, nx_out))

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }

if __name__ == "__main__":
    spec = {"output": {"grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}}}
    res = solve(spec)
    print("Done. Interp shape:", res["u"].shape, "Iters:", res["solver_info"]["iterations"])
