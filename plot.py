import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the results JSON
with open('res', 'r') as f:
    data = json.load(f)

# Build DataFrame, include consensus with mutation_prob=0
records = []
for entry in data:
    rd = entry.get('result_data', {})
    td = entry.get('task_data', {})
    if 'ari' in rd:
        algo = td.get('algorithm')
        run_type = td.get('run_type', 'base')
        # consensus tasks don't have mutation_prob, set to 0
        mp = td.get('mutation_prob', 0.0)  
        records.append({
            'algorithm': algo,
            'run_type': run_type,
            'mutation_prob': mp,
            'ari': rd['ari']
        })

df = pd.DataFrame(records)

# Separate base and consensus
base_df = df[df['run_type'] == 'base']
cons_df = df[df['run_type'] == 'consensus']

# Aggregate base: mean+std
base_agg = base_df.groupby(['mutation_prob', 'algorithm'])['ari'].agg(['mean', 'std']).reset_index()

# Aggregate consensus: mean only (flat line)
cons_agg = cons_df.groupby('algorithm')['ari'].mean().reset_index()

# Plot
fig, ax = plt.subplots()

# Base algorithms lines with error bars
for algo, grp in base_agg.groupby('algorithm'):
    ax.errorbar(
        grp['mutation_prob'],
        grp['mean'],
        yerr=grp['std'],
        marker='o',
        label=f"{algo} (base)"
    )

# Consensus algorithms as horizontal lines
for _, row in cons_agg.iterrows():
    ax.hlines(
        row['ari'],
        xmin=base_agg['mutation_prob'].min(),
        xmax=base_agg['mutation_prob'].max(),
        linestyles='--',
        label=f"{row['algorithm']} (consensus)"
    )

ax.set_xlabel('Mutation Probability')
ax.set_ylabel('Mean ARI')
ax.set_title('ARI vs Mutation Probability: Base vs Consensus')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
