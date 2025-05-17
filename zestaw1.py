import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def stworz_liste_sasiedztwa(graf):
    lista = []
    for node in sorted(graf.nodes):
        lista.append(sorted(graf.adj[node]))
    return lista

def stworz_macierz_sasiedztwa(graf):
    n = len(graf.nodes)
    # zapewnia poprawne dzialanie dla grafow o numeracji wierzcholkow
    # niezaczynajacej sie od 0
    lowest = np.min(graf.nodes)
    macierz = [[0] * n for _ in range(n)]
    for n1, n2 in sorted(graf.edges, key = lambda x: x[0]):
        macierz[n1-lowest][n2-lowest] = 1
        macierz[n2-lowest][n1-lowest] = 1
    return macierz

def stworz_macierz_incydencji(graf):
    m = len(graf.edges)
    n = len(graf.nodes)
    # zapewnia poprawne dzialanie dla grafow o numeracji wierzcholkow
    # niezaczynajacej sie od 0
    lowest = np.min(graf.nodes)
    macierz = [[0] * m for _ in range(n)]
    for i, edge in enumerate(graf.edges):
        macierz[edge[0]-lowest][i] = 1
        macierz[edge[1]-lowest][i] = 1
    return macierz

def graf_z_listy_sasiedztwa(lista_sasiedztwa, start = 1):
    n = len(lista_sasiedztwa)
    # uwzgledniamy wierzcholki izolowane
    graf = nx.Graph()
    graf.add_nodes_from(range(start, n+start))
    for i, sasiedzi in enumerate(lista_sasiedztwa):
        for sasiad in sasiedzi:  # Dla każdego sąsiada dodajemy krawędź
            graf.add_edge(i+start, sasiad)
    return graf

def graf_z_macierzy_sasiedztwa(macierz_sasiedztwa, start = 1):
    n = len(macierz_sasiedztwa)
    # uwzgledniamy wierzcholki izolowane
    graf = nx.Graph()
    graf.add_nodes_from(range(start, n+start))
    for i in range(n):
        for j in range(i + 1, n):
            if macierz_sasiedztwa[i][j] == 1:
                graf.add_edge(i + start, j + start)
    return graf

def graf_z_macierzy_incydencji(macierz_incydencji, start = 1):
    n = len(macierz_incydencji)
    # uwzgledniamy wierzcholki izolowane
    graf = nx.Graph()
    graf.add_nodes_from(range(start, n+start))
    macierz_incydencji = np.array(macierz_incydencji)
    for j in range(n):
        node1, node2 = np.where(macierz_incydencji[:, j] == 1)[0]
        graf.add_edge(node1+start, node2+start)
    return graf

def zakoduj_liste_sasiedztwa(lista_sasiedztwa):
    graf = graf_z_listy_sasiedztwa(lista_sasiedztwa)
    macierz_sasiedztwa = stworz_macierz_sasiedztwa(graf)
    macierz_incydencji = stworz_macierz_incydencji(graf)
    return graf, macierz_sasiedztwa, macierz_incydencji

def zakoduj_macierz_sasiedztwa(macierz_sasiedztwa):
    graf = graf_z_macierzy_sasiedztwa(macierz_sasiedztwa)
    lista_sasiedztwa = stworz_liste_sasiedztwa(graf)
    macierz_incydencji = stworz_macierz_incydencji(graf)
    return graf, lista_sasiedztwa, macierz_incydencji

def zakoduj_macierz_incydencji(macierz_incydencji):
    graf = graf_z_macierzy_incydencji(macierz_incydencji)
    lista_sasiedztwa = stworz_liste_sasiedztwa(graf)
    macierz_sasiedztwa = stworz_macierz_sasiedztwa(graf)
    return graf, lista_sasiedztwa, macierz_sasiedztwa

def zapisz_reprezentacje_do_pliku(reprezentacja, filename):
    """Zapisuje reprezentacje grafu do pliku."""
    with open(filename, "w") as file:
        for wiersz in reprezentacja:
            file.write(" ".join(map(str, wiersz)) + "\n")

def sprawdz_poprawnosc_listy(filename):
    """Sprawdza poprawność listy sąsiedztwa wczytanej z pliku."""
    with open(filename, "r") as file:
        lista_sasiedztwa = [list(map(int, line.split())) for line in file]

    n = len(lista_sasiedztwa)
    for node, sasiedzi in enumerate(lista_sasiedztwa,start=1):
        for sasiad in sasiedzi:
            if node not in lista_sasiedztwa[sasiad-1]:
                return False
                # raise ValueError(f"Błąd: Punkt {node} jest połączony z {sasiad}, ale {sasiad} nie jest połączony z {node}.")
    return True

def wczytaj_liste_sasiedztwa(filename):
    lista_sasiedztwa = []
    with open(filename, "r") as file:
        for line in file:
            sasiedzi = list(map(int, line.split()))
            lista_sasiedztwa.append(sasiedzi)
    return lista_sasiedztwa

def wczytaj_macierz_sasiedztwa(filename):
    with open(filename, "r") as file:
        macierz_sasiedztwa = [list(map(int, line.split())) for line in file]
    return macierz_sasiedztwa

def wczytaj_macierz_incydencji(filename):
    with open(filename, "r") as file:
        macierz_incydencji = [list(map(int, line.split())) for line in file]
    return macierz_incydencji

def sprawdz_poprawnosc_macierzy_sasiedztwa(filename):
    """Sprawdza poprawność macierzy sąsiedztwa."""
    with open(filename, "r") as file:
        macierz = [list(map(int, line.split())) for line in file]
    n = len(macierz)
    for i, wiersz in enumerate(macierz):
        if len(wiersz) != n:
            return False
            #return f"Błąd: Wiersz {i + 1} nie ma {n} elementów."
        if any(wartosc not in (0, 1) for wartosc in wiersz):
            return False
            #return f"Błąd: Wiersz {i + 1} zawiera wartości inne niż 0 i 1."
    return True
    #return "Macierz jest poprawna."

def sprawdz_poprawnosc_macierzy_incydencji(filename):
    """Sprawdza poprawność macierzy incydencji."""
    with open(filename, "r") as file:
        macierz = [list(map(int, line.split())) for line in file]
    m = len(macierz[0])
    for i, wiersz in enumerate(macierz):
        if len(wiersz) != m:
            return False
        if any(wartosc not in (0, 1) for wartosc in wiersz):
            return False
    return True

    ## Grafy losowe
def graf_krawedzi(n, k, start=1):
    """Generuje losowy graf o n punktach i k krawędziach."""
    max_k = (n * (n - 1)) // 2
    if n <= 0:
        raise ValueError("Niepoprawna wartość n. Musi być większa od 0.")
    if k < 0 or k > max_k:
        raise ValueError(f"Niepoprawna wartość k. Musi być w zakresie od 0 do {max_k} dla n={n}.")

    graph = nx.Graph()
    graph.add_nodes_from(range(start, n+start))
    wszystkie_krawedzi = [(i, j) for i in range(start, n+start) for j in range(i+start, n+start)]
    losowe_krawedzi = random.sample(wszystkie_krawedzi, k)
    # for p1, p2 in losowe_krawedzi:
    #     graph.add_edge(p1, p2)
    graph.add_edges_from(losowe_krawedzi)
    return graph

def graf_probability(n, p, start=1):
    """Generuje losowy graf o n punktach i prawdopodobieństwie p dla każdej krawędzi."""
    if n <= 0:
        raise ValueError("Niepoprawna wartość n. Musi być większa od 0.")
    if not (0 <= p <= 1):
        raise ValueError("Niepoprawna wartość p. Musi być w zakresie od 0 do 1.")

    graph = nx.Graph()
    graph.add_nodes_from(range(start, n+start))
    for i in range(start, n+start):
        for j in range(i+start, n+start):
            if random.random() < p:
                graph.add_edge(i, j)
    return graph

def rysuj_graf(graph, title="Graf", *, pos="circle", node_color="gray", edge_color="gray", edge_labels = None, show = True, save = False):
    """Narysowac graf z węzłami równomiernie rozłożonymi na okręgu."""
    #n = ilosc_wierzcholkow
    #katy = np.linspace(0, 2 * np.pi, n, endpoint=False)
    #pozycje_wierzcholkow = [(np.cos(k), np.sin(k)) for k in katy] # tablica z wartosciami x i y dla poszczegolnych wierzcholkow
    #narysowac wierzcholki i polaczyc je krawedziami z krawedzie
    n = len(graph.nodes)
    if n == 0:
        raise ValueError(f"Graf jest pusty")
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    if pos is None:
        pos = nx.spring_layout(graph)
    elif pos == "circle":
        katy_polozenia = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {wierzcholek: (np.cos(kat), np.sin(kat)) for wierzcholek, kat in zip(graph.nodes, katy_polozenia)}
        circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--')
        ax.add_artist(circle)
    nx.draw(graph, pos, with_labels=True, node_color=node_color, edge_color=edge_color,
            node_size=500, font_size=10, ax=ax)
    ax.set_title(title)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    plt.axis("equal")
    if show:
        plt.show()
    if save:
        plt.savefig(f'{title}.png')

if __name__ == "__main__":
    show_graphs = False
    save_graphs = True

    graf1 = nx.Graph()
    graf1.add_nodes_from([0, 1, 2, 3, 4])
    graf1.add_edges_from([(0, 1), (0, 3), (1, 2), (1, 3), (3, 4)])
    lista_sas1 = stworz_liste_sasiedztwa(graf1)
    for i, row in enumerate(lista_sas1):
        print(f"{i}: {row}")
    print()
    rysuj_graf(graf1, f"G1", show=show_graphs, save=save_graphs)
    zapisz_reprezentacje_do_pliku(lista_sas1, "test_lista1.txt")
    macierz_sas1 = stworz_macierz_sasiedztwa(graf1)
    for i, row in enumerate(macierz_sas1):
        print(f"{row}")
    print()
    zapisz_reprezentacje_do_pliku(macierz_sas1, "test_macierz_sas1.txt")
    macierz_inc1 = stworz_macierz_incydencji(graf1)
    for i, row in enumerate(macierz_inc1):
        print(f"{row}")
    print()
    zapisz_reprezentacje_do_pliku(macierz_inc1, "test_macierz_inc1.txt")
    # Generowanie i rysowanie wykresów
    dla_k_values = [5, 10, 20, 30, 40]
    dla_p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = 20
    for k in dla_k_values:
       graf = graf_krawedzi(n, k)
       rysuj_graf(graf,f"Graf losowy (n={n}, k={k})", show=show_graphs, save=save_graphs)

    for p in dla_p_values:
       graf = graf_probability(n, p)
       rysuj_graf(graf,f"Graf losowy (n={n}, p={p})", show=show_graphs, save=save_graphs)

    print(sprawdz_poprawnosc_macierzy_sasiedztwa("macierz.txt"))
    print(sprawdz_poprawnosc_listy("lista.txt"))

    # Tworzenie grafu z pliku
    graf_matrix = graf_z_macierzy_sasiedztwa(wczytaj_macierz_sasiedztwa("macierz.txt"))
    rysuj_graf(graf_matrix,f"Graf z pliku z macierzy sasiedztwa", show=show_graphs, save=save_graphs)

    # Tworzenie grafu z listy sąsiedztwa
    graf_lista = graf_z_listy_sasiedztwa(wczytaj_liste_sasiedztwa("lista.txt"))
    rysuj_graf(graf_lista,f"Graf z pliku z lista sasiedztwa", show=show_graphs, save=save_graphs)
    lista_z_matrix = stworz_liste_sasiedztwa(graf_matrix)
    matrix_z_listy = stworz_macierz_sasiedztwa(graf_lista)
    zapisz_reprezentacje_do_pliku(lista_z_matrix,f"nowa_lista_sasiedztwa.txt")
    zapisz_reprezentacje_do_pliku(matrix_z_listy,f"nowa_macierz_sasiedztwa.txt")

    print(sprawdz_poprawnosc_macierzy_incydencji('test_macierz_inc1.txt'))
    graf_macierz_incydencji = graf_z_macierzy_incydencji(wczytaj_macierz_incydencji('test_macierz_inc1.txt'), start=0)
    rysuj_graf(graf_macierz_incydencji, 'z incydencji', show=show_graphs, save=save_graphs)