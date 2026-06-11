"""
02_visualize_samples.py
=======================
Visualización de muestras del dataset con máscaras coloreadas.

Genera 2 figuras (train / test), cada una con 4 filas x 2 columnas
(imagen de entrada | máscara coloreada), guardadas en output/ como PNG y SVG.

Mapa de colores:
    0   → negro   (0,   0,   0  )
    1   → verde   (34,  139, 34 )
    2   → azul    (30,  144, 255)
    255 → blanco  (255, 255, 255)
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SEED = 42

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 4

# Mapa clase → color RGB (0-255)
CLASS_COLOR_MAP = {
    0:   (0,   0,   0),    # negro   → fondo
    1:   (34,  139, 34),   # verde
    2:   (30,  144, 255),  # azul
    255: (255, 255, 255),  # blanco
}

CLASS_LABELS = {
    0:   "Clase 0 – Fondo",
    1:   "Clase 1",
    2:   "Clase 2",
    255: "Clase 255",
}


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Carga imagen como array RGB."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convierte máscara de índices a imagen RGB usando CLASS_COLOR_MAP."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_val, color in CLASS_COLOR_MAP.items():
        rgb[mask == class_val] = color
    return rgb


def sample_pairs(split: str, n: int) -> list[tuple[Path, Path]]:
    """Devuelve n pares (imagen, máscara) aleatorios del split dado."""
    img_dir  = DATA_DIR / "images" / split
    mask_dir = DATA_DIR / "masks"  / split

    images = sorted(img_dir.glob("*.png"))
    assert len(images) >= n, (
        f"No hay suficientes imágenes en {img_dir} "
        f"(encontradas: {len(images)}, requeridas: {n})"
    )

    random.seed(SEED)
    selected = random.sample(images, n)
    pairs = []
    for img_path in selected:
        mask_path = mask_dir / img_path.name
        assert mask_path.exists(), f"Máscara no encontrada: {mask_path}"
        pairs.append((img_path, mask_path))
    return pairs


def build_legend():
    """Construye parches de leyenda para las clases."""
    patches = []
    for val, color in CLASS_COLOR_MAP.items():
        norm_color = tuple(c / 255 for c in color)
        patch = mpatches.Patch(color=norm_color, label=CLASS_LABELS[val])
        patches.append(patch)
    return patches


def make_figure(split: str, pairs: list[tuple[Path, Path]]) -> plt.Figure:
    """
    Crea figura de N_SAMPLES filas × 2 columnas.
    Columna izquierda: imagen de entrada.
    Columna derecha:   máscara coloreada.
    """
    fig, axes = plt.subplots(
        nrows=N_SAMPLES, ncols=2,
        figsize=(8, N_SAMPLES * 3.5),
        constrained_layout=True,
    )
    fig.suptitle(f"Muestras – {split.upper()}", fontsize=14, fontweight="bold")

    for row, (img_path, mask_path) in enumerate(pairs):
        img  = load_image(img_path)
        mask = np.array(Image.open(mask_path))
        mask_rgb = colorize_mask(mask)

        # Imagen de entrada
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(img_path.name, fontsize=7)
        axes[row, 0].axis("off")

        # Máscara coloreada
        axes[row, 1].imshow(mask_rgb)
        axes[row, 1].set_title(f"Máscara – {img_path.name}", fontsize=7)
        axes[row, 1].axis("off")

    # Leyenda al pie de la figura
    legend_patches = build_legend()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CLASS_COLOR_MAP),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    return fig


def save_figure(fig: plt.Figure, name: str):
    """Guarda la figura en PNG y SVG dentro de output/."""
    for ext in ("png", "svg"):
        out_path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", format=ext)
        print(f"  Guardado: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    for split in ("train", "test"):
        print(f"\n[{split.upper()}] Seleccionando {N_SAMPLES} muestras...")
        pairs = sample_pairs(split, N_SAMPLES)
        for img_p, msk_p in pairs:
            print(f"  {img_p.name}")

        fig = make_figure(split, pairs)
        save_figure(fig, f"samples_{split}")
        plt.close(fig)

    print("\nListo. Figuras guardadas en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
