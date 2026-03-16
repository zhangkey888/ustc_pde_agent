u_bc = fem.Function(V)
u_expr = fem.Expression(u_ex, V.element.interpolation_points())
u_bc.interpolate(u_expr)