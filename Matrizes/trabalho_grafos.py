import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class GraphAnalysisFacade:
    # INICIALIZAÇÃO
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dados = None
        self.matriz_inc = None
        self.matriz_sim = None
        self.matriz_cooc = None
        self.G_inc = None
        self.G_sim = None
        self.G_cooc = None

    # LEITURA DO DATASET
    def carregar_dados(self):
        self.dados = pd.read_csv(self.dataset_path)
        print("\n Dataset ")
        print(self.dados.head())

    # MATRIZ DE INCIDÊNCIA
    def construir_matriz_incidencia(self):
        self.matriz_inc = pd.pivot_table(
            self.dados,
            values="weight",
            index="from",
            columns="to",
            aggfunc="sum",
            fill_value=0
        )
        print("\n Matriz de Incidência ")
        print(self.matriz_inc)
        self.matriz_inc.to_csv("matriz_incidencia.csv")
        print("Arquivo salvo: matriz_incidencia.csv")

    # MATRIZ DE SIMILARIDADE
    def construir_matriz_similaridade(self):
        A = self.matriz_inc.values
        S = A @ A.T
        np.fill_diagonal(S, 0)
        self.matriz_sim = pd.DataFrame(
            S,
            index=self.matriz_inc.index,
            columns=self.matriz_inc.index
        )
        print("\n Matriz de Similaridade ")
        print(self.matriz_sim)
        self.matriz_sim.to_csv("matriz_similaridade.csv")
        print("Arquivo salvo: matriz_similaridade.csv")

    # MATRIZ DE COOCORRÊNCIA
    def construir_matriz_coocorrencia(self):
        A = self.matriz_inc.values
        C = A.T @ A
        np.fill_diagonal(C, 0)
        self.matriz_cooc = pd.DataFrame(
            C,
            index=self.matriz_inc.columns,
            columns=self.matriz_inc.columns
        )
        print("\n Matriz de Coocorrência ")
        print(self.matriz_cooc)
        self.matriz_cooc.to_csv("matriz_coocorrencia.csv")
        print("Arquivo salvo: matriz_coocorrencia.csv")

    # GRAFO DE INCIDÊNCIA
    def construir_grafo_incidencia(self):
        self.G_inc = nx.Graph()
        alunos = list(self.matriz_inc.index)
        generos = list(self.matriz_inc.columns)

        self.G_inc.add_nodes_from(alunos, tipo="aluno")
        self.G_inc.add_nodes_from(generos, tipo="genero")

        for aluno in alunos:
            for genero in generos:
                peso = self.matriz_inc.loc[aluno, genero]
                if peso > 0:
                    self.G_inc.add_edge(aluno, genero, weight=peso)

        print("\n Grafo de Incidência ")
        print(f"Nós: {self.G_inc.number_of_nodes()} | Arestas: {self.G_inc.number_of_edges()}")

    # GRAFO DE SIMILARIDADE
    def construir_grafo_similaridade(self):
        self.G_sim = nx.Graph()
        alunos = list(self.matriz_sim.index)
        self.G_sim.add_nodes_from(alunos)

        for i, a1 in enumerate(alunos):
            for j, a2 in enumerate(alunos):
                if j <= i:
                    continue
                peso = self.matriz_sim.loc[a1, a2]
                if peso > 0:
                    self.G_sim.add_edge(a1, a2, weight=peso)

        print("\n Grafo de Similaridade ")
        print(f"Nós: {self.G_sim.number_of_nodes()} | Arestas: {self.G_sim.number_of_edges()}")

    # GRAFO DE COOCORRÊNCIA
    def construir_grafo_coocorrencia(self):
        self.G_cooc = nx.Graph()
        generos = list(self.matriz_cooc.index)
        self.G_cooc.add_nodes_from(generos, tipo="genero")

        for i, g1 in enumerate(generos):
            for j, g2 in enumerate(generos):
                if j <= i:
                    continue
                peso = self.matriz_cooc.loc[g1, g2]
                if peso > 0:
                    self.G_cooc.add_edge(g1, g2, weight=peso)

        print("\n Grafo de Coocorrência ")
        print(f"Nós: {self.G_cooc.number_of_nodes()} | Arestas: {self.G_cooc.number_of_edges()}")

    # MÉTRICAS
    @staticmethod
    def metricas_grafo(G: nx.Graph, nome: str):
        print(f"\n {nome} ")
        n_nos = G.number_of_nodes()
        n_arestas = G.number_of_edges()
        densidade = nx.density(G)
        graus = dict(G.degree())
        grau_medio = sum(graus.values()) / n_nos if n_nos > 0 else 0
        grau_max = max(graus.values()) if graus else 0
        nos_maior_grau = [n for n, g in graus.items() if g == grau_max]

        print(f"Nós: {n_nos}")
        print(f"Arestas: {n_arestas}")
        print(f"Densidade: {densidade:.4f}")
        print(f"Grau médio: {grau_medio:.2f}")
        print(f"Grau máximo: {grau_max} (nós: {nos_maior_grau})")

        cent_grau = nx.degree_centrality(G)
        cent_betw = nx.betweenness_centrality(G)
        cent_close = nx.closeness_centrality(G)

        if cent_grau:
            no_cg = max(cent_grau, key=cent_grau.get)
            print(f"Maior centralidade de grau: {no_cg} ({cent_grau[no_cg]:.4f})")

        if cent_betw:
            no_cb = max(cent_betw, key=cent_betw.get)
            print(f"Maior betweenness: {no_cb} ({cent_betw[no_cb]:.4f})")

        if cent_close:
            no_cc = max(cent_close, key=cent_close.get)
            print(f"Maior closeness: {no_cc} ({cent_close[no_cc]:.4f})")

        if n_nos > 1:
            print(f"Clustering médio: {nx.average_clustering(G):.4f}")
        else:
            print("Clustering médio: não definido")

        if n_nos > 0 and nx.is_connected(G):
            print(f"Diâmetro: {nx.diameter(G)}")
            print(f"Caminho médio: {nx.average_shortest_path_length(G):.4f}")
        else:
            print("Diâmetro / caminho médio: grafo desconexo ou vazio")

    # DESENHO DOS GRAFOS
    @staticmethod
    def desenhar_grafo(G: nx.Graph, titulo: str, salvar_nome: str):
        plt.figure(figsize=(9, 7))
        pos = nx.spring_layout(G, seed=42)

        cores = []
        for n in G.nodes():
            tipo = G.nodes[n].get("tipo", "aluno")
            if tipo == "aluno":
                cores.append("#6a5acd")   
            elif tipo == "genero":
                cores.append("#ff8c00")  
            else:
                cores.append("#1f78b4")

        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_color=cores,
            node_size=900,
            font_size=9,
            edge_color="#aaaaaa"
        )

        plt.title(titulo, fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(salvar_nome, dpi=300)
        print(f"Imagem salva: {salvar_nome}")
        plt.show()

    # EXECUÇÃO 
    def executar(self, desenhar=False):
        self.carregar_dados()
        self.construir_matriz_incidencia()
        self.construir_matriz_similaridade()
        self.construir_matriz_coocorrencia()
        self.construir_grafo_incidencia()
        self.construir_grafo_similaridade()
        self.construir_grafo_coocorrencia()

        self.metricas_grafo(self.G_inc, "Grafo de Incidência (Alunos x Gêneros)")
        self.metricas_grafo(self.G_sim, "Grafo de Similaridade (Alunos)")
        self.metricas_grafo(self.G_cooc, "Grafo de Coocorrência (Gêneros)")

        if desenhar:
            self.desenhar_grafo(self.G_inc, "Grafo de Incidência", "grafo_incidencia.png")
            self.desenhar_grafo(self.G_sim, "Grafo de Similaridade", "grafo_similaridade.png")
            self.desenhar_grafo(self.G_cooc, "Grafo de Coocorrência", "grafo_coocorrencia.png")


if __name__ == "__main__":
    facade = GraphAnalysisFacade("Dataset.txt")
    facade.executar(desenhar=True)
