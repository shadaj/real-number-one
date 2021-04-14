import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_score
import sys
from os.path import basename, normpath
import glob

import os
from mip import Model, CONTINUOUS, BINARY, maximize, xsum

def solve(G: nx.Graph, max_c, max_k):
    """
    Args:
        G: networkx.Graph
    Returns:
        c: list of cities to remove
        k: list of edges to remove
    """
    model = Model()
    model.threads = int(os.getenv("THREAD_COUNT", "24"))
    model.max_mip_gap = 1e-15

    skipped_nodes = [model.add_var(var_type=BINARY) for node in G]

    flow_over_edge = [[model.add_var(var_type=CONTINUOUS) for j in G] for i in G]
    is_untakeable_edge = [[model.add_var(var_type=BINARY) for j in G] for i in G]
    # no flow over nonexistent edges
    for i in G:
        for j in G:
            if not G.has_edge(i, j):
                model += flow_over_edge[i][j] == 0
                model += is_untakeable_edge[i][j] == 1

    model += xsum(flow_over_edge[0][other_node] for other_node in G) == (len(G) - 1) - xsum(skipped_nodes)
    for other_node in G:
        model += flow_over_edge[other_node][0] == 0

    distance_to_node = [model.add_var(var_type=CONTINUOUS) for node in G]
    model += distance_to_node[0] == 0

    # in any connected subset of G, there will always be a shortest path w/ length <= sum of all edges
    # so a fake_infinity will never be better than any other option
    fake_infinity = sum([weight for _, _, weight in G.edges().data("weight")])
    skipped_edges = []
    model += skipped_nodes[0] == 0
    model += skipped_nodes[len(G) - 1] == 0

    for node_a, node_b, weight in G.edges().data("weight"):
        edge_skip_var = model.add_var(var_type=BINARY)
        skipped_edges.append((edge_skip_var, (node_a, node_b)))
        model += is_untakeable_edge[node_a][node_b] >= edge_skip_var
        model += is_untakeable_edge[node_a][node_b] >= skipped_nodes[node_a]
        model += is_untakeable_edge[node_a][node_b] >= skipped_nodes[node_b]
        model += is_untakeable_edge[node_a][node_b] <= edge_skip_var + skipped_nodes[node_a] + skipped_nodes[node_b]
        is_untakeable_edge[node_b][node_a] = is_untakeable_edge[node_a][node_b]
        is_untakeable = is_untakeable_edge[node_a][node_b]
        overall_range = 1

        # is_untakeable = edge_skip_var + skipped_nodes[node_a] + skipped_nodes[node_b]
        # overall_range = 3

        model += distance_to_node[node_a] <= distance_to_node[node_b] + weight + is_untakeable * fake_infinity # still need the term because otherwise no node can be skipped
        model += distance_to_node[node_b] <= distance_to_node[node_a] + weight + is_untakeable * fake_infinity # still need the term because otherwise no node can be skipped

        # optional, also handled by post-processing
        # model += edge_skip_var <= 1 - skipped_nodes[node_a]
        # model += edge_skip_var <= 1 - skipped_nodes[node_b]

        # no flow if edge or node removed
        model += flow_over_edge[node_a][node_b] <= (len(G) - 1) - is_untakeable * ((len(G) - 1) / overall_range)
        model += flow_over_edge[node_b][node_a] <= (len(G) - 1) - is_untakeable * ((len(G) - 1) / overall_range)

    # actually makes performance worse
    # for node in G:
    #     # immediately force the distance to infinity if we skip the node
    #     model += distance_to_node[node] >= skipped_nodes[node] * fake_infinity

    for node in G:
        if node != 0:
            flow_into_node = xsum(flow_over_edge[other_node][node] for other_node in G)
            flow_out_of_node = xsum(flow_over_edge[node][other_node] for other_node in G)
            model += flow_into_node - flow_out_of_node == 1 - skipped_nodes[node]

    model += xsum([var for var, _ in skipped_edges]) <= max_k
    model += xsum(skipped_nodes) <= max_c
    # for node in G:
    #     model += distance_to_node[node] <= fake_infinity

    if True: # don't match flow to shortest path
        model.objective = maximize(distance_to_node[len(G) - 1])
    else:
        model += xsum([
            xsum([flow_over_edge[i][j] * G[i][j]["weight"] for j in G if G.has_edge(i, j)])
            for i in G
        ]) == distance_to_node[len(G) - 1]
        model.objective = maximize(xsum([
            xsum([flow_over_edge[i][j] * G[i][j]["weight"] for j in G if G.has_edge(i, j)])
            for i in G
        ]))

    # model.solver.set_int_param("MIPFocus", 3)
    model.preprocess = 2
    status = model.optimize()
    if model.num_solutions > 0:
        c = []
        for node in G:
            if skipped_nodes[node].x > 0.99:
                c.append(node)
        k = []
        for skip_var, edge in skipped_edges:
            if skip_var.x > 0.99 and not ((edge[0] in c) or (edge[1] in c)):
                k.append(edge)
        return c, k


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
