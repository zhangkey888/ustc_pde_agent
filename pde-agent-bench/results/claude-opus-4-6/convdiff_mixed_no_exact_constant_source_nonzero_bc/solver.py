import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.005)
    beta = params.get("beta", [12.0, 0.0])
    source_term_val = pde.get("source_term_value", 1.0)
    
    bc_spec = pde.get("boundary_conditions", {})
    
    # Mesh resolution - use fine mesh for high Peclet number
    nx = ny = 128
    mesh_resolution = nx
    element_degree = 1

    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term
    f = fem.Constant(domain, ScalarType(source_term_val))
    
    # Convection velocity
    beta_vec = fem.Constant(domain, np.array(beta, dtype=ScalarType))
    
    # Diffusion coefficient
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    # Boundary conditions
    # Parse BC value - for "nonzero_bc" case
    bc_value = 0.0
    if "dirichlet" in bc_spec:
        dbc = bc_spec["dirichlet"]
        if isinstance(dbc, dict):
            bc_value = dbc.get("value", 0.0)
        elif isinstance(dbc, (int, float)):
            bc_value = float(dbc)
    
    # If bc_spec has specific boundary info
    # For mixed_type with nonzero_bc, we need to figure out the BC
    # Let's check the case spec more carefully
    bc_type = pde.get("bc_type", "nonzero_bc")
    
    # Try to get boundary condition details
    bcs_list = []
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Determine BC value from case_spec
    if "boundary_conditions" in pde:
        bc_info = pde["boundary_conditions"]
        if isinstance(bc_info, dict):
            # Could have different BCs on different boundaries
            if "dirichlet" in bc_info:
                d_info = bc_info["dirichlet"]
                if isinstance(d_info, (int, float)):
                    bc_value = float(d_info)
                elif isinstance(d_info, dict):
                    bc_value = d_info.get("value", 0.0)
            elif "value" in bc_info:
                bc_value = bc_info["value"]
    
    # For "nonzero_bc" with no exact solution, typical setup:
    # Different BCs on different boundaries
    # Common pattern: u=1 on inlet (left), u=0 on other boundaries, or similar
    # Let's check if there are specific boundary values
    
    # Parse more carefully
    bc_parts = {}
    if isinstance(bc_spec, dict):
        for key, val in bc_spec.items():
            if key == "dirichlet":
                if isinstance(val, dict):
                    # Could be {"left": 1.0, "right": 0.0, ...} or {"value": ...}
                    if "value" in val:
                        bc_value = float(val["value"])
                    else:
                        bc_parts = val
                elif isinstance(val, (int, float)):
                    bc_value = float(val)
            elif key in ["left", "right", "top", "bottom"]:
                bc_parts[key] = float(val) if isinstance(val, (int, float)) else val
    
    # If we have specific boundary parts
    if bc_parts:
        for bname, bval in bc_parts.items():
            if isinstance(bval, (int, float)):
                bval_f = float(bval)
            elif isinstance(bval, dict):
                bval_f = float(bval.get("value", 0.0))
            else:
                continue
                
            if bname == "left":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif bname == "right":
                marker = lambda x: np.isclose(x[0], 1.0)
            elif bname == "bottom":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif bname == "top":
                marker = lambda x: np.isclose(x[1], 1.0)
            else:
                continue
            
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs_list.append(fem.dirichletbc(ScalarType(bval_f), dofs, V))
    
    # If no specific parts found, apply uniform BC on all boundaries
    if not bcs_list:
        # For nonzero_bc, let's try a reasonable default
        # Common: u = some value on all boundaries
        # If bc_value is still 0, try to find it elsewhere
        if bc_value == 0.0 and "nonzero" in str(bc_type):
            bc_value = 1.0  # reasonable default for nonzero BC
        
        all_boundary = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        all_dofs = fem.locate_dofs_topological(V, fdim, all_boundary)
        bcs_list.append(fem.dirichletbc(ScalarType(bc_value), all_dofs, V))
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    
    # SUPG stabilization parameter
    tau = h / (2.0 * beta_norm) * (ufl.cosh(Pe_cell) / ufl.sinh(Pe_cell) - 1.0 / Pe_cell)
    # Simpler formula that avoids potential issues:
    # tau = h / (2 * beta_norm) * min(1, Pe_cell/3)
    # Use a simpler, more robust formula
    tau = ufl.conditional(ufl.gt(Pe_cell, 1.0), 
                          h / (2.0 * beta_norm),
                          h * h / (6.0 * eps_const))
    
    # Standard Galerkin terms
    a_gal = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_gal = f * v * ufl.dx
    
    # SUPG additional terms
    # Residual applied to trial function: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian of u is zero within elements
    # So residual ≈ beta.grad(u) - f
    R_u = ufl.dot(beta_vec, ufl.grad(u)) - f  # for P1, -eps*lap(u) = 0
    
    # SUPG test function modification
    supg_test = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(R_u + f, supg_test) * ufl.dx  # only the u-dependent part
    # Actually: a_supg contributes to both a and L
    # a += tau * (beta.grad(u)) * (beta.grad(v)) dx
    # L += tau * f * (beta.grad(v)) dx
    # (since -eps*lap(u) = 0 for P1)
    
    a_stab = tau * ufl.dot(beta_vec, ufl.grad(u)) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    L_stab = tau * f * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    
    a_total = a_gal + a_stab
    L_total = L_gal + L_stab
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=bcs_list,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
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
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }