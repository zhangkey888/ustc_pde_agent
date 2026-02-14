import dolfinx.fem
import inspect
print("Functions in dolfinx.fem:")
for name, obj in inspect.getmembers(dolfinx.fem):
    if inspect.isfunction(obj):
        print(f"  {name}")
print("\nClasses in dolfinx.fem:")
for name, obj in inspect.getmembers(dolfinx.fem):
    if inspect.isclass(obj):
        print(f"  {name}")
