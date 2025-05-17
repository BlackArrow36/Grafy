import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from zestaw1 import graf_krawedzi, stworz_liste_sasiedztwa, rysuj_graf
from zestaw2 import znajdz_skladowe


def wygeneruj_graf_spojny_wazony(n):
    if n <= 0:
        raise ValueError(f"Liczba wierzchołków ({n}) musi być dodatnia")
    while True:   
        m = random.randint(n-1, (n * (n - 1)) // 2)
        graph = graf_krawedzi(n, m)
        lista_sasiedztwa = stworz_liste_sasiedztwa(graph)
        lista_sasiedztwa_dict = {i : sasiedzi 
                                 for i, sasiedzi in enumerate(lista_sasiedztwa, start=1)}
        skladowe = znajdz_skladowe(lista_sasiedztwa_dict)
        if len(skladowe) == 1:
            for edge in graph.edges:
                graph[edge[0]][edge[1]]['weight'] = random.randint(1, 10)
            return graph

def dijkstra(graph, node):
    distances, predecessors = dijkstra_init(graph, node)
    ready_nodes = set()
    weights = {node1 : {node2 : graph[node1][node2]['weight'] 
                        for node2 in graph.adj[node1]} for node1 in graph.nodes}
    while(len(ready_nodes) != len(graph.nodes)):
        node1 = min(distances, key=lambda node : float('inf') if node in ready_nodes 
                                                else distances[node])
        ready_nodes.add(node1)
        for node2 in graph.adj[node1]:
            if node2 not in ready_nodes:
                dijkstra_relax(distances, predecessors, node1, node2, weights)
    return distances, predecessors

def dijkstra_init(graph, node):
    distances = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[node] = 0
    return distances, predecessors

def dijkstra_relax(distances, predecessors, node1, node2, weights):
    if distances[node2] > distances[node1] + weights[node1][node2]:
        distances[node2] = distances[node1] + weights[node1][node2]
        predecessors[node2] = node1

def print_shortest_paths(graph, node1, distances, predecessors):
    for node2 in graph.nodes:
        print(f'{node1} --> {node2} [{distances[node2]}] : ', end="")
        reversed_path = []
        predecessor = predecessors[node2]
        while predecessor is not None:
            reversed_path.append(predecessor)
            predecessor = predecessors[predecessor]
        for node in reversed(reversed_path):
            print(f'{node} - ', end="")
        print(f'{node2}')

def create_distances_matrix(graph):
    matrix = []
    for node in graph.nodes:
        distances, _ = dijkstra(graph, node)
        matrix.append(list(distances.values()))
    return matrix

def find_center(graph):
    distances_matrix_dict = dict(enumerate(distances_matrix, start=1))
    center = min(distances_matrix_dict, key= lambda row : sum(distances_matrix_dict[row]))
    return center, sum(distances_matrix_dict[center])

def find_minimax_center(graph):
    distances_matrix_dict = dict(enumerate(distances_matrix, start=1))
    center = min(distances_matrix_dict, key= lambda row : max(distances_matrix_dict[row]))
    return center, max(distances_matrix_dict[center])

def prim(graph):
    MST = nx.Graph()
    MST.add_node(list(graph.nodes)[0])
    W = set(list(graph.nodes)[1:])
    while len(MST.nodes) != len(graph.nodes):
        viable_edges = [edge for edge in graph.edges(data=True) 
                        if ((edge[0] in MST.nodes) == (edge[1] in W))]
        lightest_edge = sorted(viable_edges, key=
                               lambda edge : edge[2]['weight'])[0]
        MST.add_edges_from([lightest_edge])
        W.discard(lightest_edge[0])
        W.discard(lightest_edge[1])
    MST_res = nx.Graph()
    MST_res.add_nodes_from(sorted(MST.nodes))
    MST_res.add_edges_from(MST.edges(data=True))
    return MST_res


if __name__ == "__main__":
    n = 7
    graph = wygeneruj_graf_spojny_wazony(n)
    rysuj_graf(graph, edge_labels=nx.get_edge_attributes(graph, "weight"))
    node1 = 1
    distances, predecessors = dijkstra(graph, node1)
    print_shortest_paths(graph, node1, distances, predecessors)
    distances_matrix = create_distances_matrix(graph)
    for row in distances_matrix:
        print(row)
    center, distances_sum = find_center(graph)
    print(f'Graph center: node {center}, distances sum: {distances_sum}')
    center, max_distance = find_minimax_center(graph)
    print(f'Graph minimax center: node {center}, max distance: {max_distance}')
    MST = prim(graph)
    rysuj_graf(MST, edge_labels=nx.get_edge_attributes(MST, "weight"))