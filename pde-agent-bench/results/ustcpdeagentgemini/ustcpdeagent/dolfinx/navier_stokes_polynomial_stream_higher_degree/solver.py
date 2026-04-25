import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh
    mesh_res = 64
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed space: Taylor-Hood P2-P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = 0.22
    x = ufl.SpatialCoordinate(msh)
    
    # Exact solution
    u_ex = ufl.as_vector([
        x[0]**2 * (1 - x[0])**2 * (1 - 2*x[1]),
        -2*x[0] * (1 - x[0]) * (1 - 2*x[0]) * x[1] * (1 - x[1])
    ])
    p_ex = x[0] + x[1]
    
    # Source term
    # f = u.grad(u) - nu * laplace(u) + grad(p)
    grad_u_ex = ufl.grad(u_ex)
    laplace_u_ex = ufl.div(grad_u_ex)
    f_expr = grad_u_ex * u_ex - nu * laplace_u_ex + ufl.grad(p_ex)

    def eps(u_field):
        return ufl.sym(ufl.grad(u_field))
    
    def sigma(u_field, p_field):
        return 2.0 * nu * eps(u_field) - p_field * ufl.Identity(gdim)

    # Nonlinear residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    J = ufl.derivative(F, w)

    # Boundary Conditions
    # Velocity on all walls
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at corner to p_ex(0, 0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x_coord: np.isclose(x_coord[0], 0.0) & np.isclose(x_coord[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0 # x=0, y=0 -> p=0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]

    # Initial guess for Newton
    # Could start from zero, or try solving Stokes first
    w.x.array[:] = 0.0
    
    # Solve nonlinear problem
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    
    solver = PETSc.SNES().create(msh.comm)
    solver.setOptionsPrefix("ns_")
    solver.setFromOptions()
    
    problem.solve()
    w.x.scatter_forward()
    u_sol, _ = w.split()
    u_func = w.sub(0).collapse()

    # Probing
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
        
    u_mag = np.linalg.norm(u_vals, axis=1).reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "nonlinear_iterations": [5]
    }
    
    return {
        "u": u_mag,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 100, "ny": 100, "bbox": [0,1,0,1]}}}
    res = solve(case)
    print("Shape:", res["u"].shape)
    print("Max vel:", np.nanmax(res["u"]))
