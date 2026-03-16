x, u_ex_y))
    
    # Source term f
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex))
    
    # Boundary Conditions
    u_bc_func = fem.Function(V)
    u_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_expr)
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    def origin_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), origin_marker)
    
    bcs = [bc_u]
    if len(dofs_p[0]) > 0