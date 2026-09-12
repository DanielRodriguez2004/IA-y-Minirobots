"""
AC probabilístico 2D para difusión de una enfermedad
Basado en la sección 2.7 de las diapositivas de Autómatas Celulares.

Modelo:
    Estados:
      Np  = vacío
      Ps  = persona sana/susceptible
      Cs  = persona enferma sintomática
      Cns = persona enferma asintomática
      Ct  = persona en cuarentena/cuidados médicos
      Pac = persona recuperada con anticuerpos
      M   = persona muerta (NO se representa en el retículo)

    Vecindad:
      Moore (8 vecinos + célula central).

    Reglas adaptadas de la diapositiva de automatas celulares:
      - Una persona sana puede infectarse según el número de vecinos
        infecciosos (Cs + Cns).
      - Se usa una probabilidad base de 0.10 por vecino infeccioso:
            P(infección | n vecinos) = 1 - (1 - 0.10)^n
      - La diapositiva da 0.4 para Cns y 0.7 para Cs.
        Como esos valores no suman 1, aquí se interpretan como pesos:
            P(Cns | nueva infección) = 0.4 / (0.4 + 0.7)
            P(Cs  | nueva infección) = 0.7 / (0.4 + 0.7)
      - Cs -> Ct después de 7 ciclos.
      - Cns -> Pac después de 30 ciclos.
      - Ct -> Pac con prob. 0.8 o M con prob. 0.2 después de 20 ciclos.
      - Después de las transiciones, las personas se pueden desplazar
        N/S/E/O con probabilidad 0.25, solo si la celda destino está vacía.
      - La actualización es paralela: el nuevo estado se calcula a partir
        del estado anterior.

Dependencias:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


# 1. Parámetros del experimento


FILAS = 100
COLUMNAS = 100
CICLOS = 150

PERSONAS_SANA = 5000
SINTOMATICOS_INICIALES = 5
ASINTOMATICOS_INICIALES = 5

PROB_POR_VECINO = 0.10

# Pesos tomados de la diapositiva: 0.4 y 0.7.
PESO_ASINTOMATICO = 0.4
PESO_SINTOMATICO = 0.7

P_CNS = PESO_ASINTOMATICO / (PESO_ASINTOMATICO + PESO_SINTOMATICO)
P_CS = PESO_SINTOMATICO / (PESO_ASINTOMATICO + PESO_SINTOMATICO)

PROB_RECUPERACION_CT = 0.8
PROB_MUERTE_CT = 0.2

CICLOS_CS = 7
CICLOS_CNS = 30
CICLOS_CT = 20

SEMILLA = 42

# Códigos de estados del retículo.
VACIO = 0
SANO = 1
SINTOMATICO = 2
ASINTOMATICO = 3
CUARENTENA = 4
RECUPERADO = 5



# 2. Creación del retículo y configuración inicial


def crear_reticulo():
    """Crea C0 con personas distribuidas aleatoriamente."""
    rng = np.random.default_rng(SEMILLA)

    total = FILAS * COLUMNAS
    requeridas = PERSONAS_SANA + SINTOMATICOS_INICIALES + ASINTOMATICOS_INICIALES

    if requeridas > total:
        raise ValueError("Hay más personas iniciales que celdas disponibles.")

    grid = np.zeros((FILAS, COLUMNAS), dtype=np.int8)

    posiciones = rng.choice(total, size=requeridas, replace=False)

    grid.flat[posiciones[:PERSONAS_SANA]] = SANO
    inicio = PERSONAS_SANA
    fin = inicio + SINTOMATICOS_INICIALES
    grid.flat[posiciones[inicio:fin]] = SINTOMATICO
    grid.flat[posiciones[fin:fin + ASINTOMATICOS_INICIALES]] = ASINTOMATICO

    # "Edad" del estado: cuántos ciclos lleva una persona en Cs/Cns/Ct.
    edad = np.zeros_like(grid, dtype=np.int16)

    return grid, edad, rng



# 3. Vecindad de Moore


def vecinos_moore(a):
    
    resultado = np.zeros_like(a, dtype=np.int16)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue

            desplazada = np.zeros_like(a, dtype=np.int16)

            filas_origen = slice(max(0, -di), min(FILAS, FILAS - di))
            cols_origen = slice(max(0, -dj), min(COLUMNAS, COLUMNAS - dj))

            filas_destino = slice(max(0, di), min(FILAS, FILAS + di))
            cols_destino = slice(max(0, dj), min(COLUMNAS, COLUMNAS + dj))

            desplazada[filas_destino, cols_destino] = a[filas_origen, cols_origen]
            resultado += desplazada

    return resultado



# 4. Regla probabilística de infección


def probabilidad_infeccion(n_vecinos):
    
    return 1.0 - (1.0 - PROB_POR_VECINO) ** n_vecinos



# 5. Movimiento N/S/E/O


def mover_personas(grid, edad, rng):
   
    nuevo_grid = grid.copy()
    nueva_edad = edad.copy()

    personas = np.argwhere(grid != VACIO)
    rng.shuffle(personas)

    direcciones = np.array([
        (-1, 0),  # N
        (1, 0),   # S
        (0, 1),   # E
        (0, -1),  # O
    ])

    for i, j in personas:
        if nuevo_grid[i, j] == VACIO:
            continue

        # Probabilidad total de intentar movimiento = 1.
        # La dirección elegida es uniforme: 0.25 cada una.
        di, dj = direcciones[rng.integers(0, 4)]
        ni, nj = i + di, j + dj

        if 0 <= ni < FILAS and 0 <= nj < COLUMNAS:
            if nuevo_grid[ni, nj] == VACIO:
                nuevo_grid[ni, nj] = nuevo_grid[i, j]
                nueva_edad[ni, nj] = nueva_edad[i, j]
                nuevo_grid[i, j] = VACIO
                nueva_edad[i, j] = 0

    return nuevo_grid, nueva_edad



# 6. Una actualización completa del AC


def paso(grid, edad, muertos, rng):
    """Aplica una iteración t -> t+1 del AC."""
    
    # A. Reglas de transición sanitaria en paralelo
    
    siguiente = grid.copy()
    siguiente_edad = np.zeros_like(edad)

    infecciosos = ((grid == SINTOMATICO) | (grid == ASINTOMATICO)).astype(np.int8)
    n_infecciosos = vecinos_moore(infecciosos)

    sanos = (grid == SANO)
    p_inf = probabilidad_infeccion(n_infecciosos)

    infectar = sanos & (n_infecciosos > 0) & (rng.random(grid.shape) < p_inf)

    # Tipo de la nueva infección.
    selector = rng.random(grid.shape)
    nuevas_cns = infectar & (selector < P_CNS)
    nuevas_cs = infectar & ~nuevas_cns

    siguiente[nuevas_cns] = ASINTOMATICO
    siguiente[nuevas_cs] = SINTOMATICO

    # Las personas sintomáticas pasan a cuarentena luego de 7 ciclos.
    pasar_ct = (grid == SINTOMATICO) & (edad + 1 >= CICLOS_CS)
    siguiente[pasar_ct] = CUARENTENA

    # Las asintomáticas se recuperan luego de 30 ciclos.
    pasar_recuperado = (grid == ASINTOMATICO) & (edad + 1 >= CICLOS_CNS)
    siguiente[pasar_recuperado] = RECUPERADO

    # Las personas en Ct, después de 20 ciclos:
    en_ct = (grid == CUARENTENA) & (edad + 1 >= CICLOS_CT)
    aleatorio_ct = rng.random(grid.shape)

    recuperar_ct = en_ct & (aleatorio_ct < PROB_RECUPERACION_CT)
    morir_ct = en_ct & ~recuperar_ct

    siguiente[recuperar_ct] = RECUPERADO
    siguiente[morir_ct] = VACIO

    muertos += int(np.sum(morir_ct))

    
    # B. Actualizar edades según el nuevo estado
    
    siguiente_edad[(siguiente == SINTOMATICO) & (grid == SINTOMATICO)] = edad[
        (siguiente == SINTOMATICO) & (grid == SINTOMATICO)
    ] + 1

    siguiente_edad[(siguiente == ASINTOMATICO) & (grid == ASINTOMATICO)] = edad[
        (siguiente == ASINTOMATICO) & (grid == ASINTOMATICO)
    ] + 1

    siguiente_edad[(siguiente == CUARENTENA) & (grid == CUARENTENA)] = edad[
        (siguiente == CUARENTENA) & (grid == CUARENTENA)
    ] + 1

    # Entrada a Cs/Cns/Ct: empiezan con contador 0.
    siguiente_edad[(siguiente != grid) & (siguiente == SINTOMATICO)] = 0
    siguiente_edad[(siguiente != grid) & (siguiente == ASINTOMATICO)] = 0
    siguiente_edad[(siguiente != grid) & (siguiente == CUARENTENA)] = 0

    
    # C. Movimiento N/S/E/O
    
    siguiente, siguiente_edad = mover_personas(siguiente, siguiente_edad, rng)

    return siguiente, siguiente_edad, muertos



# 7. Conteo de estados


def contar(grid):
    return {
        "Vacío": int(np.sum(grid == VACIO)),
        "Sanos": int(np.sum(grid == SANO)),
        "Sintomáticos": int(np.sum(grid == SINTOMATICO)),
        "Asintomáticos": int(np.sum(grid == ASINTOMATICO)),
        "Cuarentena": int(np.sum(grid == CUARENTENA)),
        "Recuperados": int(np.sum(grid == RECUPERADO)),
    }



# 8. Simulación completa


def simular():
    grid, edad, rng = crear_reticulo()

    historia = []
    configuraciones = []
    muertos = 0

    historia.append(contar(grid))
    configuraciones.append(grid.copy())

    for _ in range(CICLOS):
        grid, edad, muertos = paso(grid, edad, muertos, rng)

        historia.append(contar(grid))
        configuraciones.append(grid.copy())

    return grid, historia, configuraciones, muertos





grid_final, historia, configuraciones, muertos_final = simular()

print("\n========= RESULTADOS DEL AC =========")
print(f"Ciclos simulados: {CICLOS}")
print(f"Muertos acumulados: {muertos_final}")
print("\nEstado final:")
for nombre, valor in historia[-1].items():
    print(f"{nombre:16s}: {valor}")



ciclos = np.arange(len(historia))

plt.figure(figsize=(10, 6))

for clave in ["Sanos", "Sintomáticos", "Asintomáticos",
              "Cuarentena", "Recuperados"]:
    valores = [h[clave] for h in historia]
    plt.plot(ciclos, valores, label=clave)

plt.xlabel("Ciclo")
plt.ylabel("Número de celdas/personas")
plt.title("Evolución de la enfermedad - AC probabilístico 2D")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()





cmap = ListedColormap([
    "white",        # vacío
    "lightgreen",   # sano
    "red",          # sintomático
    "orange",       # asintomático
    "purple",       # cuarentena
    "deepskyblue",  # recuperado
])


from matplotlib.patches import Patch

etiquetas_estados = ["Vacío", "Sano", "Sintomático", "Asintomático", "Cuarentena", "Recuperado"]
colores_estados = ["white", "lightgreen", "red", "orange", "purple", "deepskyblue"]

elementos_leyenda = [
    Patch(facecolor=colores_estados[i], edgecolor="gray", label=etiquetas_estados[i])
    for i in range(len(etiquetas_estados))
]

plt.figure(figsize=(8, 8))
plt.imshow(grid_final, cmap=cmap, interpolation="nearest")
plt.title("Configuración final del AC")
plt.axis("off")
plt.legend(handles=elementos_leyenda, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()




MOSTRAR_ANIMACION = True

if MOSTRAR_ANIMACION:
    fig, ax = plt.subplots(figsize=(7, 7))

    imagen = ax.imshow(
        configuraciones[0],
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=5
    )

    ax.set_axis_off()
    ax.legend(handles=elementos_leyenda, loc="center left", bbox_to_anchor=(1, 0.5))
    titulo = ax.set_title("Ciclo 0")

    def actualizar(frame):
        imagen.set_data(configuraciones[frame])
        titulo.set_text(f"Ciclo {frame}")
        return imagen, titulo

    anim = FuncAnimation(
        fig,
        actualizar,
        frames=len(configuraciones),
        interval=100,
        blit=True,
        repeat=False
    )

    plt.show()

