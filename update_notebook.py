import json

with open('Fast_Introduction.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'las_noises_ratios = []' in source:
            source = source.replace(
"""las_noises_ratios = []
noise_levels = [0, 5, 10, 20, 50, 100, 200, 500]
for nl in noise_levels:
    if nl == 0:
        J_pred = job_instance_creation(w_true, D)
    else:
        J_pred, _ = generate_massive_noises(w_true, D, nl)
    r = LAS_energy_ratio(J_true, J_pred, epsilon, alpha, dt)
    las_noises_ratios.append(r)""",
"""import numpy as np
las_noises_ratios_mean = []
las_noises_ratios_std = []
noise_levels = [0, 5, 10, 20, 50, 100, 200, 500]
num_trials = 30  # number of experiments to run for each noise level
for nl in noise_levels:
    trial_ratios = []
    for _ in range(num_trials):
        if nl == 0:
            J_pred = job_instance_creation(w_true, D)
        else:
            J_pred, _ = generate_massive_noises(w_true, D, nl)
        r = LAS_energy_ratio(J_true, J_pred, epsilon, alpha, dt)
        trial_ratios.append(r)
    las_noises_ratios_mean.append(np.mean(trial_ratios))
    las_noises_ratios_std.append(np.std(trial_ratios))"""
            )
            
            source = source.replace(
                'ax1.plot(noise_levels, las_noises_ratios, marker="o", label="LAS Competitive Ratio")',
                'ax1.errorbar(noise_levels, las_noises_ratios_mean, yerr=las_noises_ratios_std, marker="o", capsize=3, label="LAS Competitive Ratio (Mean $\\\\pm$ 1$\\\\sigma$)")'
            )
            
            source = source.replace(
                'values = [OA, AVR, las_noises_ratios[0], las_noises_ratios[noise_levels.index(50)], las_noises_ratios[noise_levels.index(500)], las_blind, las_inv]',
                'values = [OA, AVR, las_noises_ratios_mean[0], las_noises_ratios_mean[noise_levels.index(50)], las_noises_ratios_mean[noise_levels.index(500)], las_blind, las_inv]'
            )

            # Split back into lines keeping the newline characters
            new_source = []
            lines = source.split('\\n')
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    new_source.append(line + '\\n')
                elif line:
                    new_source.append(line)
            cell['source'] = new_source
            cell['outputs'] = []

with open('Fast_Introduction.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
