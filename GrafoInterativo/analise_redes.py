import pandas as pd
import networkx as nx
from pyvis.network import Network
from tabulate import tabulate
import os
import webbrowser


ARQUIVO_ENTRADA = 'jogos_dataset.csv'
ARQUIVO_SAIDA = 'grafo_interativo.html'

if not os.path.exists(ARQUIVO_ENTRADA):
    print(f"ERRO: O arquivo '{ARQUIVO_ENTRADA}' não está nesta pasta!")
    exit()

print(f"Lendo dados de: {ARQUIVO_ENTRADA}...")
df = pd.read_csv(ARQUIVO_ENTRADA)

matriz = pd.pivot_table(df, values='weight', index='from', columns='to', fill_value=0)

print("\n" + "="*60)
print("             MATRIZ DE INCIDÊNCIA")
print("="*60)
try:
    print(tabulate(matriz, headers='keys', tablefmt='fancy_grid', numalign="center"))
except ImportError:
    print("Aviso: Biblioteca 'tabulate' não instalada. Imprimindo tabela simples.")
    print(matriz)
print("\n")

G = nx.from_pandas_edgelist(df, source='from', target='to', create_using=nx.DiGraph())

net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)


for node in G.nodes():
    if node in matriz.columns:
        net.add_node(node, label=node, color='#FFA500', title='Jogo', size=25, shape='box') 
    else:
        net.add_node(node, label=node, color='#00BFFF', title='Aluno', size=15) 

for source, target in G.edges():
    net.add_edge(source, target, color='#555555')

net.show_buttons(filter_=['physics'])

print("Gerando arquivo HTML...")
net.save_graph(ARQUIVO_SAIDA)

caminho_completo = os.path.join(os.getcwd(), ARQUIVO_SAIDA)
print("\n" + "="*60)
print(f"O gráfico foi criado.")
print(f"Arquivo: {caminho_completo}")
print("="*60)

print("entando abrir no navegador...")
try:
    webbrowser.open('file://' + caminho_completo)
except:
    print("Não abriu sozinho. Vá na pasta e clique em 'grafo_interativo.html'")