import networkx
import random

vertices = 50
print(vertices)

G = networkx.erdos_renyi_graph(vertices, 0.5)
assert networkx.is_connected(G)
for node_a, node_b in G.edges():
  print(f"{node_a} {node_b} {random.uniform(0, 100):.3f}")