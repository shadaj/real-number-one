import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

import os
from mip import Model, CONTINUOUS, BINARY, maximize, xsum
from mip.constants import OptimizationStatus

def solve(G: nx.Graph, max_c, max_k, timeout, existing_solution, target_distance):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    model = Model()
    model.threads = int(os.getenv("THREAD_COUNT", "24"))
    model.max_mip_gap = 1e-12

    skipped_nodes = [model.add_var(var_type=BINARY) for node in G]

    flow_over_edge = [[model.add_var(var_type=CONTINUOUS) for j in G] for i in G]

    # no flow over nonexistent edges
    for i in G:
        for j in G:
            if not G.has_edge(i, j):
                model += flow_over_edge[i][j] == 0

    model += xsum(flow_over_edge[0][other_node] for other_node in G) == (len(G) - 1) - xsum(skipped_nodes)
    for other_node in G:
        model += flow_over_edge[other_node][0] == 0

    distance_to_node = [model.add_var(var_type=CONTINUOUS) for node in G]
    model += distance_to_node[0] == 0

    # in any connected subset of G, there will always be a shortest path w/ length <= sum of all edges
    # so a fake_infinity will never be better than any other option
    fake_infinity = sum([weight for _, _, weight in G.edges().data("weight")])
    skipped_edges = []
    skipped_edge_map = {}
    model += skipped_nodes[0] == 0
    model += skipped_nodes[len(G) - 1] == 0

    for node_a, node_b, weight in G.edges().data("weight"):
        edge_skip_var = model.add_var(var_type=BINARY)
        skipped_edges.append((edge_skip_var, (node_a, node_b)))
        model += edge_skip_var >= skipped_nodes[node_a]
        model += edge_skip_var >= skipped_nodes[node_b]

        model += distance_to_node[node_a] <= distance_to_node[node_b] + weight + edge_skip_var * fake_infinity
        model += distance_to_node[node_b] <= distance_to_node[node_a] + weight + edge_skip_var * fake_infinity

        # no flow if edge or node removed
        model += flow_over_edge[node_a][node_b] <= (len(G) - 1) - edge_skip_var * (len(G) - 1)
        model += flow_over_edge[node_b][node_a] <= (len(G) - 1) - edge_skip_var * (len(G) - 1)

    # results in binary variable leakage
    for node in G:
        # immediately force the distance to infinity if we skip the node
        model += distance_to_node[node] >= skipped_nodes[node] * fake_infinity
        model += distance_to_node[node] <= fake_infinity

    for node in G:
        if node != 0:
            flow_into_node = xsum(flow_over_edge[other_node][node] for other_node in G)
            flow_out_of_node = xsum(flow_over_edge[node][other_node] for other_node in G)
            model += flow_into_node - flow_out_of_node == 1 - skipped_nodes[node]

    model += xsum([var for var, _ in skipped_edges]) - xsum([skipped_nodes[i] * len(G[i]) for i in G]) <= max_k
    model += xsum(skipped_nodes) <= max_c

    if target_distance:
        model += distance_to_node[len(G) - 1] >= target_distance
        model += distance_to_node[len(G) - 1] <= target_distance

    model.objective = maximize(distance_to_node[len(G) - 1])
    # these cuts are used more often but aggressively using them doesn't seem to help
    model.solver.set_int_param("FlowCoverCuts", 2)
    model.solver.set_int_param("MIRCuts", 2)

    if existing_solution:
        solution_variables = []
        for node in G:
            solution_variables.append((skipped_nodes[node], 1.0 if node in existing_solution[0] else 0.0))
        for edge_var, edge in skipped_edges:
            is_skipped = (edge in existing_solution[1]) or ((edge[1], edge[0]) in existing_solution[1]) or \
                         (edge[0] in existing_solution[0]) or (edge[1] in existing_solution[0])
            solution_variables.append((edge_var, 1.0 if is_skipped else 0.0))
        model.start = solution_variables

    status = model.optimize(max_seconds=timeout)
    if model.num_solutions > 0:
        c = []
        for node in G:
            if skipped_nodes[node].x > 0.99:
                c.append(node)
        k = []
        for skip_var, edge in skipped_edges:
            if skip_var.x > 0.99 and not ((edge[0] in c) or (edge[1] in c)):
                k.append(edge)
        return c, k, status == OptimizationStatus.OPTIMAL, model.gap
    else:
        return None


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 4
    path = sys.argv[1]
    max_c = int(sys.argv[2])
    max_k = int(sys.argv[3])
    G = read_input_file(path)
    c, k = solve(G, max_c, max_k)
    assert is_valid_solution(G, c, k)
    print("Shortest Path Difference: {}".format(calculate_score(G, c, k)))
    write_output_file(G, c, k, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G = read_input_file(input_path)
#         c, k = solve(G)
#         assert is_valid_solution(G, c, k)
#         distance = calculate_score(G, c, k)
#         write_output_file(G, c, k, output_path)
