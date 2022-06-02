import random
import time as tim
import numpy as np

datos = np.loadtxt('ch130.tsp', max_rows=130)
datos2 = np.loadtxt('a280.tsp', max_rows=280)
ch130 = np.delete(datos, 0, 1)
a280 = np.delete(datos2, 0, 1)


def inicializar_matrices(datos, feromona_inicial):
    f, c = datos.shape
    matriz_heuristica = np.zeros((f, f))
    matriz_distancias = np.zeros((f, f))
    matriz_feromonas = np.ones((f, f))
    matriz_feromonas *= feromona_inicial
    for i in range(0, f):
        a = np.array((datos[i][0], datos[i][1]))
        for j in range(i, f):
            b = np.array((datos[j][0], datos[j][1]))
            if i != j:
                dist = int(np.linalg.norm(a - b))
                distkm = dist
                if dist == 0:
                    dist = 1
                else:
                    dist = 1 / dist
            else:
                dist = np.inf
                distkm = np.inf
            matriz_heuristica[i][j] = dist
            matriz_heuristica[j][i] = dist
            matriz_distancias[i][j] = distkm
            matriz_distancias[j][i] = distkm
    # tengo que calcular la distancia de cada punto con el resto(matriz de adyaciencia)
    return matriz_heuristica, matriz_feromonas, matriz_distancias


def calcular_sumatorio(fila_heuristica, fila_feromonas, tabu):
    denominador = 0
    for cont, (heu, fer) in enumerate(zip(fila_heuristica, fila_feromonas)):
        if heu != np.inf and cont not in tabu:
            denominador += heu * fer
    return denominador


def calcular_fila_probabilidades(fila_heuristica, fila_feromonas, tabu):
    denominador = calcular_sumatorio(fila_heuristica, fila_feromonas, tabu)
    fila_probabilidad = []
    for cont, (heu, fer) in enumerate(zip(fila_heuristica, fila_feromonas)):  # comprobar que funciona el enumerate
        if heu != np.inf and cont not in tabu:

            numerador = heu * fer
            fila_probabilidad.append((numerador / denominador))
        else:
            fila_probabilidad.append(-1)
    return fila_probabilidad


def seleccion_ciudad(fila_probabilidades):
    valor = random.uniform(0.0, 1.0)
    acumulado = 0
    posicion = -5
    for pos, prob in enumerate(fila_probabilidades):
        if prob > 0:
            acumulado += prob
            if acumulado > valor:
                posicion = pos
                fila_probabilidades[pos] = -1
    return posicion


def evaporacion_feromonas(matriz_feromonas, coeficiente_evaporacion, matriz_aporte):
    p = coeficiente_evaporacion
    f, c = matriz_feromonas.shape
    for fila in range(f):
        for columna in range(c):
            valor_actualizado = (1 - p) * matriz_feromonas[fila][columna] + matriz_aporte[fila][columna]
            matriz_feromonas[fila][columna] = valor_actualizado
            matriz_feromonas[columna][fila] = valor_actualizado
    return matriz_feromonas


# def greedy():


def coste(matriz_distancia, camino):
    coste = 0
    for i in range(len(camino) - 1):
        j = i + 1
        coste += matriz_distancia[i][j]
    return coste


def calcular_matriz_aporte(costes, f):
    matriz_aporte = np.zeros((f, f))
    for c in costes:
        aporte = 1 / c[0]
        camino = c[1]
        for i in range(0, f - 1):

            matriz_aporte[camino[i]][camino[i + 1]] += aporte
            matriz_aporte[camino[i + 1]][camino[i]] += aporte
        matriz_aporte[camino[0]][camino[f-1]] += aporte
        matriz_aporte[camino[f-1]][camino[0]] += aporte
    return matriz_aporte


def generar_solucion_hormiga(nodo_inicial, matriz_heuristica, matriz_feromonas, matriz_distancia):
    nodo_inicial = int(nodo_inicial)
    tabu = []
    tabu.append(nodo_inicial)
    # fila_heuristica, fila_feromonas, tabu):
    f, c = matriz_heuristica.shape
    ciudad_actual = nodo_inicial
    max = int(f + nodo_inicial)
    for i in range(nodo_inicial, max - 1):
        i = i % f
        fila_probabilidades = calcular_fila_probabilidades(matriz_heuristica[ciudad_actual],
                                                           matriz_feromonas[ciudad_actual], tabu)
        ciudad_actual = seleccion_ciudad(fila_probabilidades)
        tabu.append(ciudad_actual)
    return tabu


def sh(datos):
    num_hormigas = 10
    alpha = 1
    beta = 2
    p = 0.1
    n = 280
      # cambiar por la de la greedy con el coste y el vector
    mejor_solucion=list([10000000000, np.arange(0,279)])

    # c_greedy=greedy
    # usar la alpha y beta
    f, c = datos.shape
    L = np.zeros((num_hormigas, f),dtype=int)
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromona_inicial=10)
    iter = 0
    inicio = tim.time()
    while (iter < (10000 * f)) and ((tim.time() - inicio) < 60):
        # print(tim.time()-inicio)
        for i in range(num_hormigas):
            L[i][0] = random.randint(0, f)
        costes = []

        for k in range(0, num_hormigas):
            L[k]=generar_solucion_hormiga(L[k][0], matriz_heuristica, matriz_feromonas, matriz_distancia)
            costes.append(list([coste(matriz_distancia, L[k]), L[k]]))

        costes = sorted(costes, key=lambda x: (x[0]))
        mejor_actual = costes[0]
        matriz_aporte = calcular_matriz_aporte(costes, f)

        matriz_feromonas = evaporacion_feromonas(matriz_feromonas, p, matriz_aporte)
        print(mejor_solucion[0])
        if mejor_actual[0] < mejor_solucion[0]:
            mejor_solucion = mejor_actual
    return mejor_solucion


print(sh(a280))
