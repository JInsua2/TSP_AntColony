import random
import time
import time as tim

import matplotlib.pyplot as plt
import numpy as np

datos = np.loadtxt('ch130.tsp', max_rows=130)
datos2 = np.loadtxt('a280.tsp', max_rows=280)
ch130 = np.delete(datos, 0, 1)
a280 = np.delete(datos2, 0, 1)

camino_optimo130 = np.loadtxt('ch130.opt.tour', max_rows=130, dtype=int)
camino_optimo280 = np.loadtxt('a280.opt.tour', max_rows=280, dtype=int)


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


def calcular_sumatorio(fila_heuristica, fila_feromonas, tabu, alpha, beta):
    denominador = 0
    for cont, (heu, fer) in enumerate(zip(fila_heuristica, fila_feromonas)):
        if heu != np.inf and cont not in tabu:
            denominador += (heu ** beta) * (fer ** alpha)
    return denominador


def calcular_fila_probabilidades(fila_heuristica, fila_feromonas, tabu, alpha, beta):
    denominador = calcular_sumatorio(fila_heuristica, fila_feromonas, tabu, alpha, beta)
    fila_probabilidad = []
    for cont, (heu, fer) in enumerate(zip(fila_heuristica, fila_feromonas)):  # comprobar que funciona el enumerate
        if heu != np.inf and cont not in tabu:

            numerador = (heu ** beta) * (fer ** alpha)
            fila_probabilidad.append((numerador / denominador))
        else:
            fila_probabilidad.append(-1)
    return fila_probabilidad


def calcular_fila_probabilidades_colonia(fila_heuristica, fila_feromonas, tabu, alpha, beta):
    fila_probabilidad = []
    for cont, (heu, fer) in enumerate(zip(fila_heuristica, fila_feromonas)):  # comprobar que funciona el enumerate
        if heu != np.inf and cont not in tabu:

            valor = (heu ** beta) * (fer ** alpha)
            fila_probabilidad.append(valor)
        else:
            fila_probabilidad.append(-1)
    return fila_probabilidad


def seleccion_ciudad(fila_probabilidades):
    valor = random.uniform(0.0, 1.0)
    # valor=0.673
    acumulado = 0
    posicion = -5
    # print("len de la fila prob",len(fila_probabilidades))
    for pos, prob in enumerate(fila_probabilidades):

        if prob > 0:
            acumulado += prob
            if pos == 280:
                print("la longitud cuando da fallo es ", len(fila_probabilidades), " ----- ", pos)
            if acumulado > valor:
                posicion = pos
                fila_probabilidades[pos] = -1
                break
    return posicion


def seleccion_ciudad_colonia(fila_probabilidades, matriz_feromonas):
    max_inicial = fila_probabilidades[0]
    posicion = 0
    # print("len de la fila prob",len(fila_probabilidades))
    for pos, prob in enumerate(fila_probabilidades):
        if prob > max_inicial:
            posicion = pos
            max_inicial = prob

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


def evaporacion_feromonas_global(matriz_feromonas, coeficiente_evaporacion, matriz_aporte, camino):
    p = coeficiente_evaporacion

    for i in range(len(camino) - 1):
        valor_actualizado = (1 - p) * matriz_feromonas[camino[i]][camino[i + 1]] + matriz_aporte[camino[i]][
            camino[i + 1]]
        matriz_feromonas[camino[i]][camino[i + 1]] = valor_actualizado
        matriz_feromonas[camino[i + 1]][camino[i]] = valor_actualizado
    valor_actualizado = (1 - p) * matriz_feromonas[camino[-1]][camino[0]] + matriz_aporte[camino[i]][camino[i + 1]]
    matriz_feromonas[camino[-1]][camino[0]] = valor_actualizado
    matriz_feromonas[camino[0]][camino[-1]] = valor_actualizado

    return matriz_feromonas


def coste(matriz_distancia, camino):
    coste = 0
    for i in range(len(camino) - 1):
        j = i + 1
        coste += matriz_distancia[camino[i]][camino[j]]
    coste += matriz_distancia[camino[-1]][camino[0]]
    return coste


def calcular_matriz_aporte(costes, f):
    matriz_aporte = np.zeros((f, f))
    for c in costes:
        aporte = 1 / c[0]
        camino = c[1]
        for i in range(0, f - 1):
            matriz_aporte[camino[i]][camino[i + 1]] += aporte
            matriz_aporte[camino[i + 1]][camino[i]] += aporte
        matriz_aporte[camino[0]][camino[f - 1]] += aporte
        matriz_aporte[camino[f - 1]][camino[0]] += aporte
    return matriz_aporte


def calcular_matriz_aporte_elitista(costes, f, mejor_camino, coste_mejor, e):
    matriz_aporte = np.zeros((f, f))
    for c in costes:
        aporte = 1 / c[0]
        camino = c[1]
        for i in range(0, f - 1):
            matriz_aporte[camino[i]][camino[i + 1]] += aporte
            matriz_aporte[camino[i + 1]][camino[i]] += aporte
        matriz_aporte[camino[0]][camino[f - 1]] += aporte
        matriz_aporte[camino[f - 1]][camino[0]] += aporte
    for i in range(0, f - 1):
        matriz_aporte[mejor_camino[i]][mejor_camino[i + 1]] += e * 1 / coste_mejor
        matriz_aporte[mejor_camino[i + 1]][mejor_camino[i]] += e * 1 / coste_mejor
    matriz_aporte[mejor_camino[-1]][mejor_camino[0]] += e * 1 / coste_mejor
    matriz_aporte[mejor_camino[0]][mejor_camino[-1]] += e * 1 / coste_mejor
    return matriz_aporte


def calcular_matriz_aporte_global(camino, aporte, f):
    matriz_aporte = np.zeros((f, f))
    for i in range(0, f - 1):
        matriz_aporte[camino[i]][camino[i + 1]] += aporte
        matriz_aporte[camino[i + 1]][camino[i]] += aporte
    matriz_aporte[camino[0]][camino[f - 1]] += aporte
    matriz_aporte[camino[f - 1]][camino[0]] += aporte
    return matriz_aporte


def greedy(matriz_distancias):
    solicion_greedy = []
    f, c = matriz_distancias.shape
    nodo_actual = 0
    minimo = 1000
    solicion_greedy.append(nodo_actual)
    for i in range(1, f):
        nodo_actual = mas_cercano(matriz_distancias[nodo_actual], solicion_greedy)
        solicion_greedy.append(nodo_actual)

    return solicion_greedy


def mas_cercano(fila_distancias, tabu):
    minimo = 1000000

    for pos, valor in enumerate(fila_distancias):
        if valor != np.inf and pos not in tabu:
            if valor < minimo:
                minimo = valor
                posicion = pos
    return posicion


def generar_solucion_hormiga(nodo_inicial, matriz_heuristica, matriz_feromonas, matriz_distancia, alpha, beta):
    nodo_inicial = int(nodo_inicial)
    tabu = []
    tabu.append(nodo_inicial)
    # fila_heuristica, fila_feromonas, tabu):
    f, c = matriz_heuristica.shape
    ciudad_actual = nodo_inicial
    max = int(f + nodo_inicial)
    for x in range(len(matriz_heuristica) - 1):
        # i = i % f
        fila_probabilidades = calcular_fila_probabilidades(matriz_heuristica[ciudad_actual],
                                                           matriz_feromonas[ciudad_actual], tabu, alpha, beta)
        ciudad_actual = seleccion_ciudad(fila_probabilidades)
        nuevo_tabu = ciudad_actual
        tabu.append(nuevo_tabu)
    return tabu


def generar_solucion_hormiga_colonia(nodo_inicial, matriz_heuristica, matriz_feromonas, matriz_distancia, alpha, beta,
                                     phi, t_0):
    nodo_inicial = int(nodo_inicial)
    tabu = []
    q0 = 0.98
    tabu.append(nodo_inicial)
    # fila_heuristica, fila_feromonas, tabu):
    f, c = matriz_heuristica.shape
    ciudad_actual = nodo_inicial
    ciudad_anterior = ciudad_actual
    max = int(f + nodo_inicial)
    for x in range(len(matriz_heuristica) - 1):
        # i = i % f
        if random.uniform(0.0, 1.0 <= q0):
            fila_probabilidades = calcular_fila_probabilidades_colonia(matriz_heuristica[ciudad_actual],
                                                                       matriz_feromonas[ciudad_actual], tabu, alpha,
                                                                       beta)
            ciudad_actual = seleccion_ciudad_colonia(fila_probabilidades, matriz_feromonas)
            valor = (1 - phi) * matriz_feromonas[ciudad_anterior][ciudad_actual] + phi * t_0
            matriz_feromonas[ciudad_anterior][ciudad_actual] = valor
            matriz_feromonas[ciudad_actual][ciudad_anterior] = valor
            ciudad_anterior = ciudad_actual
            tabu.append(ciudad_actual)
        else:
            fila_probabilidades = calcular_fila_probabilidades(matriz_heuristica[ciudad_actual],
                                                               matriz_feromonas[ciudad_actual], tabu, alpha, beta)
            ciudad_actual = seleccion_ciudad_colonia(fila_probabilidades, matriz_feromonas)
            valor = (1 - phi) * matriz_feromonas[ciudad_anterior][ciudad_actual] + phi * t_0
            matriz_feromonas[ciudad_anterior][ciudad_actual] = valor
            matriz_feromonas[ciudad_actual][ciudad_anterior] = valor
            ciudad_anterior = ciudad_actual
            tabu.append(ciudad_actual)
    return tabu, matriz_feromonas


def sh(datos):
    np.random.seed(21334)
    random.seed(21334)
    # np.random.seed(732123)
    # random.seed(732123)

    num_hormigas = 10
    alpha = 1
    beta = 2
    p = 0.1
    n = 280
    x = []
    y = []
    iteracion_final = 0
    # cambiar por la de la greedy con el coste y el vector
    mejor_solucion = list([10000000000, np.arange(0, 279)])
    # c_greedy=greedy
    # usar la alpha y beta
    f, c = datos.shape
    L = np.zeros((num_hormigas, f), dtype=int)
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromona_inicial=1)
    sol_greedy = greedy(matriz_distancia)

    feromonas_inicial = 1 / (f * coste(matriz_distancia, sol_greedy))
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromonas_inicial)

    iter = 0
    inicio = tim.time()
    while ((tim.time() - inicio) < 60 * 5):
        # print(tim.time()-inicio)
        for i in range(num_hormigas):
            L[i][0] = random.randint(0, f - 1)
            # L[i][0] = 0
        costes = []
        inicio2 = tim.time()
        for k in range(0, num_hormigas):
            L[k] = generar_solucion_hormiga(L[k][0], matriz_heuristica, matriz_feromonas, matriz_distancia, alpha, beta)
            costes.append(list([coste(matriz_distancia, L[k]), L[k]]))

        costes = sorted(costes, key=lambda x: (x[0]))
        mejor_actual = costes[0]
        matriz_aporte = calcular_matriz_aporte(costes, f)
        matriz_feromonas = evaporacion_feromonas(matriz_feromonas, p, matriz_aporte)
        # print(mejor_solucion[0])
        print(tim.time() - inicio, " segundos")
        # print(iter, " iteraciones")
        if mejor_actual[0] < mejor_solucion[0]:
            mejor_solucion = mejor_actual
            iteracion_final = iter
        x.append(iter)
        y.append(mejor_solucion[0])
        iter += 1

    # plt.plot(x, y, linewidth=2.5, color='#A09BE7')
    # plt.title('Evolucion coste en sh ')
    # plt.show()
    return mejor_solucion, iteracion_final


def she(datos, num_elitistas):
    np.random.seed(21334)
    random.seed(21334)
    # np.random.seed(732123)
    # random.seed(732123)

    num_hormigas = 10
    alpha = 1
    beta = 2
    p = 0.1
    n = 280
    iteracion_final = 0
    x = []
    y = []
    mejor_solucion = list([10000000000, np.arange(0, 279)])
    f, c = datos.shape
    L = np.zeros((num_hormigas, f), dtype=int)
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromona_inicial=1)
    sol_greedy = greedy(matriz_distancia)

    feromonas_inicial = 1 / (f * coste(matriz_distancia, sol_greedy))
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromonas_inicial)

    iter = 0
    inicio = tim.time()
    while ((tim.time() - inicio) < 60 * 0.1):
        # print(tim.time()-inicio)
        for i in range(num_hormigas):
            L[i][0] = random.randint(0, f - 1)
            # L[i][0] = 0
        costes = []
        inicio2 = tim.time()
        for k in range(0, num_hormigas):
            L[k] = generar_solucion_hormiga(L[k][0], matriz_heuristica, matriz_feromonas, matriz_distancia, alpha, beta)
            costes.append(list([coste(matriz_distancia, L[k]), L[k]]))

        costes = sorted(costes, key=lambda x: (x[0]))
        mejor_actual = costes[0]

        matriz_aporte = calcular_matriz_aporte_elitista(costes, f, mejor_actual[1], mejor_actual[0], num_elitistas)

        matriz_feromonas = evaporacion_feromonas(matriz_feromonas, p, matriz_aporte)
        # print(mejor_solucion[0])
        print(tim.time() - inicio, " segundos")
        # print(iter, " iteraciones")
        if mejor_actual[0] < mejor_solucion[0]:
            mejor_solucion = mejor_actual
            iteracion_final = iter
        x.append(iter)
        y.append(mejor_solucion[0])
        iter += 1
    # plt.plot(x, y, linewidth=2.5, color='#1EB8AE')
    # plt.title('Evolucion coste en SHE ')
    # plt.show()
    return mejor_solucion, iteracion_final


def sch(datos):
    np.random.seed(21334)
    random.seed(21334)
    # np.random.seed(732123)
    # random.seed(732123)

    num_hormigas = 10
    alpha = 1
    beta = 2
    phi = 0.1
    p = 0.1
    n = 280
    iteracion_final = 0
    x = []
    y = []
    # cambiar por la de la greedy con el coste y el vector
    mejor_solucion = list([10000000000, np.arange(0, 279)])
    # c_greedy=greedy
    # usar la alpha y beta
    f, c = datos.shape
    L = np.zeros((num_hormigas, f), dtype=int)
    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromona_inicial=1)
    sol_greedy = greedy(matriz_distancia)

    feromonas_inicial = 1 / (f * coste(matriz_distancia, sol_greedy))

    matriz_heuristica, matriz_feromonas, matriz_distancia = inicializar_matrices(datos, feromonas_inicial)

    iter = 0
    inicio = tim.time()
    while ((tim.time() - inicio) < 60 * 5):
        # print(tim.time()-inicio)
        for i in range(num_hormigas):
            L[i][0] = random.randint(0, f - 1)
            # L[i][0] = 0
        costes = []
        inicio2 = tim.time()
        for k in range(0, num_hormigas):
            L[k], matriz_feromonas = generar_solucion_hormiga_colonia(L[k][0], matriz_heuristica, matriz_feromonas,
                                                                      matriz_distancia, alpha, beta, phi,
                                                                      feromonas_inicial)
            costes.append(list([coste(matriz_distancia, L[k]), L[k]]))

            costes = sorted(costes, key=lambda x: (x[0]))
            mejor_actual = costes[0]
            aporte = p / costes[0][0]
            # print("el coste es: ",costes[0][0])
            if mejor_actual[0] < mejor_solucion[0]:
                mejor_solucion = mejor_actual
                iteracion_final = iter
            matriz_aporte = calcular_matriz_aporte_global(mejor_solucion[1], aporte, f)

            matriz_feromonas = evaporacion_feromonas_global(matriz_feromonas, p, matriz_aporte, mejor_solucion[1])
            x.append(iter)
            y.append(mejor_solucion[0])
            iter += 1
            print(tim.time() - inicio, " segundos")
    # plt.plot(x, y, linewidth=2.5, color='#E5D352')
    # plt.title('Evolucion coste en SCH ')
    # plt.show()
    return mejor_solucion, iteracion_final


def dibujar_camino_vs_optimo(puntos, camino, camino_optimo):
    x = []
    y = []
    x_optimo = []
    y_optimo = []
    for c in camino:
        x.append(puntos[c][0])
        y.append(puntos[c][1])

    for c_optimo in camino_optimo:
        c_optimo += -1
        x_optimo.append(puntos[c_optimo][0])
        y_optimo.append(puntos[c_optimo][1])
    plt.plot(x, y, linewidth=2.5, color='#E5D352')
    plt.title('Camino SCH')
    plt.show()
    plt.plot(x_optimo, y_optimo, linewidth=2.5, color='black')
    plt.title('Camino Optimo')
    plt.show()
    plt.plot(x, y, linewidth=2.5, color='#E5D352', label='camino')
    plt.plot(x_optimo, y_optimo, linewidth=2.5, color='#59594A', label='camino optimo')
    plt.title('Comparacion con el optimo')
    plt.legend()
    plt.show()


parametro_datos = ch130
camino, iteracion_final = she(parametro_datos, 10)
print("-----------------------------------\n")
print("coste de la solucion ", camino[0])
print("numero de evaluaciones ", iteracion_final * 10)
print("iteracion de la ultima mejora ", iteracion_final)
dibujar_camino_vs_optimo(parametro_datos, camino[1], camino_optimo130)
