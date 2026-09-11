import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from pathlib import Path


# ============================================================
# DIAGRAMAS DE VORONOI - IPIALES
# Droguerías, centros de salud y colegios
# ============================================================

CARPETA = Path(__file__).parent

puntos_salud = np.array([
    [657, 252],
    [742, 260],
    [646, 303],
    [812, 334],
    [1161, 344],
    [1448, 373],
    [1469, 380],
    [1432, 422],
    [700, 422],
    [696, 467],
    [800, 486],
    [752, 509],
    [807, 549],
    [836, 579]
], dtype=float)


puntos_droguerias = np.array([
    [1042, 86],
    [1018, 124],
    [787, 207],
    [760, 259],
    [780, 245],
    [826, 296],
    [780, 361],
    [800, 374],
    [823, 399],
    [791, 411],
    [823, 425],
    [849, 428],
    [874, 405],
    [933, 481],
    [961, 490],
    [989, 458],
    [1070, 517],
    [1097, 526],
    [1095, 253]
], dtype=float)


puntos_colegios = np.array([
    [902, 103],
    [702, 190],
    [671, 254],
    [652, 293],
    [845, 284],
    [988, 306],
    [740, 364],
    [882, 366],
    [750, 438],
    [779, 472],
    [848, 474],
    [914, 486],
    [942, 496],
    [879, 533],
    [1298, 457],
    [1403, 541],
    [737, 597]
], dtype=float)

tamano_referencia = {
    "hospital.jpg": (1675, 829),
    "drogueria.jpg": (1776, 829),
    "colegios.jpg": (1678, 796)
}

def ajustar_puntos(puntos, imagen, tam_ref):

    alto, ancho = imagen.shape[:2]

    ancho_ref, alto_ref = tam_ref

    puntos_ajustados = puntos.copy()

    puntos_ajustados[:, 0] *= ancho / ancho_ref
    puntos_ajustados[:, 1] *= alto / alto_ref

    return puntos_ajustados


# Función para generar un diagrama de Voronoi

def generar_voronoi(
        nombre_imagen,
        puntos,
        titulo,
        archivo_salida
):

    ruta_imagen = CARPETA / nombre_imagen

    # Verificar que exista la imagen
    if not ruta_imagen.exists():
        print(f"ERROR: No se encontró {nombre_imagen}")
        return

    # Leer imagen
    imagen = plt.imread(ruta_imagen)

    # Adaptar las coordenadas al tamaño real de la imagen
    puntos = ajustar_puntos(
        puntos,
        imagen,
        tamano_referencia[nombre_imagen]
    )

    # Calcular el diagrama de Voronoi
    vor = Voronoi(puntos)

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))

    # Mostrar mapa como fondo
    ax.imshow(imagen)

    # Dibujar líneas del Voronoi
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_width=2
    )

    # Dibujar puntos generadores
    ax.scatter(
        puntos[:, 0],
        puntos[:, 1],
        s=50,
        marker="x",
        label="Establecimientos"
    )

    # Mantener exactamente los límites de la imagen
    alto, ancho = imagen.shape[:2]

    ax.set_xlim(0, ancho)
    ax.set_ylim(alto, 0)

    ax.set_title(
        titulo,
        fontsize=16
    )

    ax.legend()

    ax.axis("off")

    plt.tight_layout()

    # Guardar resultado
    ruta_salida = CARPETA / archivo_salida

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Imagen guardada en: {ruta_salida}")


# GENERACIÓN DE LOS TRES DIAGRAMAS

generar_voronoi(
    "hospital.jpg",
    puntos_salud,
    "Diagrama de Voronoi - Centros de atención de salud",
    "voronoi_salud.png"
)


generar_voronoi(
    "drogueria.jpg",
    puntos_droguerias,
    "Diagrama de Voronoi - Droguerías",
    "voronoi_droguerias.png"
)


generar_voronoi(
    "colegios.jpg",
    puntos_colegios,
    "Diagrama de Voronoi - Colegios",
    "voronoi_colegios.png"
)