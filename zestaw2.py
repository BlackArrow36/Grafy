import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#zad1
def sprawdz(stopnie):
    stopnie = sorted(stopnie, reverse=True)
    while True:
        if sum(x % 2 for x in stopnie) % 2 == 1:
            return False
        if all(x == 0 for x in stopnie):
            return True
        if stopnie[0] > len(stopnie) or any(x < 0 for x in stopnie):
            return False
        for i in range(1, stopnie[0] + 1):
            stopnie[i] -= 1
        stopnie[0] = 0
        stopnie.sort(reverse=True)

#zad2
def components_R(nr, v, G, comp):
    for u in  G[v]:
        if comp[u] == -1:
            comp[u] = nr
            components_R(nr, u, G, comp)

def components(G):
    nr=0
    comp = {v: -1 for v in G}
    nr = 0
    for v in G:
        if comp[v] == -1:
            nr += 1
            comp[v] = nr
            components_R(nr, v, G, comp)
    return comp

#Zad3
def znajdz_skladowe(G):
    visited = set()
    skladowe = []

    for v in G:
        if v not in visited:
            skladowa = DFS(G, v, visited)
            skladowe.append(skladowa)
    return skladowe

def DFS(G, v, visited):
    stack = [v]
    skladowa = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            skladowa.append(node)
            for neighbor in G[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    return sorted(skladowa)

def wypisz_skladowe(skladowe):
    print(len(skladowe))
    for i, skladowa in enumerate(skladowe, start=1):
        print(f"{i}) {' '.join(map(str, skladowa))}")

    max_index = max(range(len(skladowe)), key=lambda i: len(skladowe[i])) + 1
    print(f"\nNajwieksza skladowa ma numer {max_index}")

#zad5
def generuj_graf_k_regularny(n, k):
    if n<=k:
        return
    if k % 2 ==1 & n % 2 ==0:
        return
    G = {i: set() for i in range(n)}
    stopnie = {i: 0 for i in range(n)}
    wierzcholki = list(range(n))*k
    random.shuffle(wierzcholki)
    while wierzcholki:
        v = wierzcholki.pop()
        u = wierzcholki.pop()
        while u==v or u in G[v]:
            wierzcholki.pop()
            random.shuffle(wierzcholki)
            u = wierzcholki.pop()

        G[v].add(u)
        G[u].add(v)


    return G


def rysuj_graf(G):
    plt.figure(figsize=(6, 6))
    G_nx = nx.Graph()

    for v, sasiedzi in G.items():
        for u in sasiedzi:
            G_nx.add_edge(v, u)

    nx.draw(G_nx, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=10)
    plt.show()

if __name__ == "__main__":
    stopnie = [1, 3, 2, 3, 2, 4, 1]
    print(sprawdz(stopnie))

    stopnie=[1,3,3,4,2,3,1]
    print(sprawdz(stopnie))

    G = {
        1: [2, 5],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [1, 4, 6],
        6: [5, 7],
        7: [6, 11],
        8: [9, 10],
        9: [8, 10],
        10: [8, 9],
        11: [7]
    }
    rysuj_graf(G)

    skladowe = znajdz_skladowe(G)
    wypisz_skladowe(skladowe)
    n = 10
    k = 3

    G = generuj_graf_k_regularny(n, k)
    rysuj_graf(G)