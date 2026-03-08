with open('solver.py', 'r') as f:
    lines = f.readlines()

# Find the configurations section
for i in range(len(lines)):
    if "configurations = [" in lines[i]:
        start_idx = i
        # Find the end of the list
        for j in range(i, len(lines)):
            if "]" in lines[j]:
                end_idx = j
                break
        
        # Replace with proper configurations
        new_configs = '''    # Configurations to try in order of increasing accuracy (and time)
    # Adjusted to use full time budget
    configurations = [
        (64, 1),    # Baseline
        (128, 1),   # Better accuracy
        (128, 2),   # Even better accuracy
        (192, 2),   # Higher resolution
        (128, 3),   # Higher degree
        (192, 3),   # Both high resolution and degree
    ]
'''
        lines = lines[:start_idx] + new_configs.splitlines(keepends=True) + lines[end_idx+1:]
        break

with open('solver.py', 'w') as f:
    f.writelines(lines)
