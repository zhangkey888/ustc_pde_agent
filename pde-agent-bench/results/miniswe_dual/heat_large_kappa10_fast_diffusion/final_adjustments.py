with open('solver.py', 'r') as f:
    lines = f.readlines()

# Adjust the time margin check to be more conservative
for i in range(len(lines)):
    if "if elapsed > 0.7 * time_limit:" in lines[i]:
        lines[i] = '        if elapsed > 0.65 * time_limit:  # Leave 35% margin for evaluation\n'
        break

# Adjust the adaptive dt logic to be more conservative
for i in range(len(lines)):
    if "# If we have plenty of time, use smaller dt for better accuracy" in lines[i]:
        # Find and replace this block
        for j in range(i, i+10):
            if "dt = t_end / n_steps" in lines[j]:
                # Replace the whole block
                lines[i:j+1] = '''        # Adaptive time stepping: use smaller dt only if we're sure we have time
        dt = dt_suggested
        elapsed_so_far = time.time() - start_time
        time_remaining = time_limit - elapsed_so_far
        configs_remaining = len(configurations) - config_idx
        
        # Only reduce dt if we have significant time remaining AND it's one of the last configs
        # Be conservative: halving dt doubles solve time
        if time_remaining > 6.0 and configs_remaining == 1:  # Only for the very last configuration
            dt = dt_suggested / 2.0
        
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # Adjust to exactly reach t_end
'''.splitlines(keepends=True)
                break
        break

with open('solver.py', 'w') as f:
    f.writelines(lines)
