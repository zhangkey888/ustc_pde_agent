)

    u_values = np.full((points_array.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

nx, ny = 50, 50
x_coords = np.linspace(0, 1, nx)
y_coords = np.linspace(0, 1, ny)
X, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
points = np.vstack((X.flatten(), Y_grid.flatten(), np.zeros_like(X.flatten())))

u_values = probe_points(u_sol, points, domain)
u_grid = u_values.reshape((nx, ny))
```
Wait, what if the evaluator expects `u` to be the `fem.Function`?
In previous similar tasks, if the docstring says `numpy array with shape (nx, ny)`, it strictly expects a numpy array. I will return the numpy array.

Let's consider the solver. The problem is a mixed formulation, which results in a saddle-point-like system, though it's symmetric and not strictly a saddle