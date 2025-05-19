import tsplib95
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from time import time

def geo_para_radianos(coord):
    graus = int(coord)
    minutos = coord - graus
    return math.pi * (graus + 5.0 * minutos / 3.0) / 180.0

def distancia_geo(coord1, coord2):
    raio_terra = 6378.388
    lat1 = geo_para_radianos(coord1[0])
    lon1 = geo_para_radianos(coord1[1])
    lat2 = geo_para_radianos(coord2[0])
    lon2 = geo_para_radianos(coord2[1])
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    return int(raio_terra * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)

def carregar_problema(caminho_arquivo):
    problema = tsplib95.load(caminho_arquivo)
    nos = list(problema.get_nodes())
    n = len(nos)
    matriz_distancias = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if problema.edge_weight_type == 'EUC_2D':
                dx = problema.node_coords[i + 1][0] - problema.node_coords[j + 1][0]
                dy = problema.node_coords[i + 1][1] - problema.node_coords[j + 1][1]
                matriz_distancias[i][j] = math.hypot(dx, dy)
            elif problema.edge_weight_type == 'GEO':
                matriz_distancias[i][j] = distancia_geo(
                    problema.node_coords[i + 1], problema.node_coords[j + 1])
            elif problema.edge_weight_type == 'ATT':
                dx = problema.node_coords[i + 1][0] - problema.node_coords[j + 1][0]
                dy = problema.node_coords[i + 1][1] - problema.node_coords[j + 1][1]
                rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                tij = round(rij)
                matriz_distancias[i][j] = tij if tij >= rij else tij + 1
            else:
                matriz_distancias[i][j] = problema.get_weight(i + 1, j + 1)
    return nos, matriz_distancias

def populacao_inicial(tamanho, num_cidades):
    return [random.sample(range(num_cidades), num_cidades) for _ in range(tamanho)]

def avaliacao(trajeto, matriz_distancias):
    return sum(matriz_distancias[trajeto[i]][trajeto[(i + 1) % len(trajeto)]] for i in range(len(trajeto)))

def selecao_torneio(populacao, avaliacoes, k=3):
    selecionados = random.sample(list(zip(populacao, avaliacoes)), k)
    return min(selecionados, key=lambda x: x[1])[0]

def cruzamento(pai1, pai2):
    inicio, fim = sorted(random.sample(range(len(pai1)), 2))
    filho = [-1] * len(pai1)
    filho[inicio:fim+1] = pai1[inicio:fim+1]
    restante = [gene for gene in pai2 if gene not in filho]
    j = 0
    for i in range(len(filho)):
        if filho[i] == -1:
            filho[i] = restante[j]
            j += 1
    return filho

def mutacao(trajeto, taxa=0.02):
    for i in range(len(trajeto)):
        if random.random() < taxa:
            j = random.randint(0, len(trajeto) - 1)
            trajeto[i], trajeto[j] = trajeto[j], trajeto[i]
    return trajeto

def algoritmo_genetico(matriz_distancias, geracoes=1000, tamanho_populacao=100):
    num_cidades = len(matriz_distancias)
    populacao = populacao_inicial(tamanho_populacao, num_cidades)
    melhor_distancia = float('inf')
    melhor_geracao = 0
    convergencia = []

    for ger in range(geracoes):
        avaliacoes = [avaliacao(ind, matriz_distancias) for ind in populacao]
        melhor_da_geracao = min(avaliacoes)
        convergencia.append(melhor_da_geracao)

        if melhor_da_geracao < melhor_distancia:
            melhor_distancia = melhor_da_geracao
            melhor_geracao = ger

        nova_populacao = []
        for _ in range(tamanho_populacao):
            p1 = selecao_torneio(populacao, avaliacoes)
            p2 = selecao_torneio(populacao, avaliacoes)
            filho = mutacao(cruzamento(p1, p2))
            nova_populacao.append(filho)

        populacao = nova_populacao

    return melhor_distancia, melhor_geracao, convergencia

def executar_instancia(nome_arquivo):
    print(f'\n🗂️  Executando para: {nome_arquivo}')
    nos, matriz = carregar_problema(nome_arquivo)
    inicio = time()
    melhor_dist, geracao, convergencia = algoritmo_genetico(matriz, geracoes=1000, tamanho_populacao=150)
    tempo_total = time() - inicio
    print(f'✅ Melhor distância: {int(melhor_dist)} encontrada na geração {geracao}')
    print(f'⏱️ Tempo: {tempo_total:.2f} segundos')
    return nome_arquivo, int(melhor_dist), geracao, round(tempo_total, 2), convergencia

if __name__ == "__main__":
    arquivos = ['burma14.tsp', 'att48.tsp', 'gr202.tsp']
    otimos_conhecidos = {
        'burma14.tsp': 3323,
        'att48.tsp': 10628,
        'gr202.tsp': 40160
    }

    resultados = []

    for arquivo in arquivos:
        resultado = executar_instancia(arquivo)
        resultados.append(resultado)

    for nome_arquivo, _, _, _, convergencia in resultados:
        plt.plot(convergencia, label=nome_arquivo)

    plt.title("Convergência do Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Distância")
    plt.legend()
    plt.grid(True)
    plt.savefig("grafico_convergencia.png")
    plt.clf()

    nomes = [r[0] for r in resultados]
    tempos = [r[3] for r in resultados]

    plt.bar(nomes, tempos, color='teal')
    plt.title("Tempo de Execução por Instância")
    plt.ylabel("Tempo (s)")
    plt.savefig("grafico_tempos.png")
    plt.clf()
