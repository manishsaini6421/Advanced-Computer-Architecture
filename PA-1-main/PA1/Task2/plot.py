import matplotlib.pyplot as plt
import numpy as np

embedding_labels = ['(1M, 128)', '(1M, 512)', '(2M, 512)', '(2.5M, 256)', '(5M, 64)', '(5M, 256)', '(7M, 64)', '(6777779, 32)', '(34779331, 32)', '(6799, 777)']

software_prefetching = [1.23003, 1.2346, 1.09965, 1.25053, 1.55952, 1.2008, 1.50575, 1.81818, 1.8250, 1.0464]
simd = [2.00521, 3.47333, 3.00641, 2.45455, 1.8069, 2.17455, 1.72368, 1.63934, 1.4129, 3.58857]
combined = [2.63699, 3.76173, 3.39855, 3.09375, 3.35897, 2.90291, 2.70103, 2.70270, 2.7375, 3.70501]

x = np.arange(len(embedding_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - width, software_prefetching, width, label='Software prefetching', color='green')
bars2 = ax.bar(x, simd, width, label='SIMD', color='grey')
bars3 = ax.bar(x + width, combined, width, label='Software prefetching + SIMD', color='orange')


ax.set_ylabel('Normalized speedup\n(no optimization)')
# ax.set_ylim([1.0, 1.35])
ax.legend(fontsize=8, markerscale=0.3, frameon=False)
ax.set_xticks(x)
ax.set_xticklabels(embedding_labels)
ax.set_xlabel('(embedding table size, embedding dimension)')
ax.legend()

plt.tight_layout()
plt.savefig('plot.png')
plt.show()