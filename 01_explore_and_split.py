"""
01_explore_and_split.py
=======================
Exploración del dataset VHR y división 70/15/15 (train/val/test).

Estructura de salida:
    data/
        images/train/   images/val/   images/test/
        masks/train/    masks/val/    masks/test/
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SEED = 42

X_DIR = Path(r"C:\Drive\DATA\dataset_VHR\X_png")
Y_DIR = Path(r"C:\Drive\DATA\dataset_VHR\Y_png")

# Root del proyecto: carpeta donde vive este script
PROJECT_ROOT = Path(__file__).parent

OUTPUT_DIRS = {
    "images": {
        "train": PROJECT_ROOT / "data" / "images" / "train",
        "val":   PROJECT_ROOT / "data" / "images" / "val",
        "test":  PROJECT_ROOT / "data" / "images" / "test",
    },
    "masks": {
        "train": PROJECT_ROOT / "data" / "masks" / "train",
        "val":   PROJECT_ROOT / "data" / "masks" / "val",
        "test":  PROJECT_ROOT / "data" / "masks" / "test",
    },
}

# Fracción de muestras a analizar para clases (costoso en datasets grandes)
# Usa 1.0 para analizar todo el dataset.
CLASS_SAMPLE_FRACTION = 1.0


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def make_output_dirs():
    for group in OUTPUT_DIRS.values():
        for path in group.values():
            path.mkdir(parents=True, exist_ok=True)


def collect_pairs():
    """Devuelve lista de (x_path, y_path) con nombre coincidente."""
    x_files = sorted(X_DIR.glob("tile_*.png"))
    pairs = []
    missing = []
    for x in x_files:
        y = Y_DIR / x.name
        if y.exists():
            pairs.append((x, y))
        else:
            missing.append(x.name)
    if missing:
        print(f"  [AVISO] {len(missing)} imágenes sin máscara correspondiente: {missing[:5]} ...")
    return pairs


def image_stats(path: Path):
    """Devuelve (H, W, C, dtype, min, max) de una imagen."""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    return h, w, c, img.dtype, img.min(), img.max()


def explore_dataset(pairs):
    print("\n" + "=" * 60)
    print("EXPLORACIÓN DEL DATASET")
    print("=" * 60)

    total = len(pairs)
    print(f"\nTotal de pares imagen/máscara encontrados: {total}")

    # --- Muestreo para estadísticas ---
    n_sample = max(1, int(total * CLASS_SAMPLE_FRACTION))
    sample_pairs = random.sample(pairs, n_sample) if n_sample < total else pairs

    # Dimensiones de imágenes de entrada
    print("\n--- Imágenes de entrada (X) ---")
    x_dims = Counter()
    x_dtypes = set()
    x_ranges = []
    for x, _ in sample_pairs:
        h, w, c, dt, mn, mx = image_stats(x)
        x_dims[(h, w, c)] += 1
        x_dtypes.add(str(dt))
        x_ranges.append((int(mn), int(mx)))

    print(f"  Dimensiones (H, W, C) encontradas:")
    for dim, count in x_dims.most_common():
        print(f"    {dim}  →  {count} imágenes")
    print(f"  Tipos de dato:  {x_dtypes}")
    all_mins = [r[0] for r in x_ranges]
    all_maxs = [r[1] for r in x_ranges]
    print(f"  Rango de valores: min={min(all_mins)}, max={max(all_maxs)}")

    # Dimensiones de máscaras de salida
    print("\n--- Máscaras de salida (Y) ---")
    y_dims = Counter()
    y_dtypes = set()
    all_classes = set()
    for _, y in sample_pairs:
        h, w, c, dt, _, _ = image_stats(y)
        y_dims[(h, w, c)] += 1
        y_dtypes.add(str(dt))
        mask = np.array(Image.open(y))
        all_classes.update(np.unique(mask).tolist())

    print(f"  Dimensiones (H, W, C) encontradas:")
    for dim, count in y_dims.most_common():
        print(f"    {dim}  →  {count} máscaras")
    print(f"  Tipos de dato:  {y_dtypes}")
    print(f"  Clases únicas detectadas ({len(all_classes)}): {sorted(all_classes)}")
    print(f"\n  >> NUM_CLASSES recomendado para el modelo: {len(all_classes)}")

    return len(all_classes), sorted(all_classes)


def split_and_copy(pairs):
    """División 70/15/15 y copia a carpetas de destino."""
    print("\n" + "=" * 60)
    print("DIVISIÓN Y COPIA DE ARCHIVOS")
    print("=" * 60)

    # Primera partición: 70% train, 30% rest
    train_pairs, rest_pairs = train_test_split(
        pairs, test_size=0.30, random_state=SEED, shuffle=True
    )
    # Segunda partición: 50% de rest → val (15%), 50% → test (15%)
    val_pairs, test_pairs = train_test_split(
        rest_pairs, test_size=0.50, random_state=SEED, shuffle=True
    )

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    for split_name, split_pairs in splits.items():
        print(f"\n  [{split_name.upper()}]  {len(split_pairs)} pares")
        for x_src, y_src in split_pairs:
            shutil.copy2(x_src, OUTPUT_DIRS["images"][split_name] / x_src.name)
            shutil.copy2(y_src, OUTPUT_DIRS["masks"][split_name]  / y_src.name)

    total = len(pairs)
    for split_name, split_pairs in splits.items():
        pct = len(split_pairs) / total * 100
        print(f"    {split_name:6s}: {len(split_pairs):>5}  ({pct:.1f}%)")

    return splits


def print_output_structure():
    print("\n--- Estructura de carpetas generada ---")
    for group_name, group in OUTPUT_DIRS.items():
        for split_name, path in group.items():
            n = len(list(path.glob("*.png")))
            print(f"  {path.relative_to(PROJECT_ROOT)}  →  {n} archivos")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Verificando rutas de entrada...")
    assert X_DIR.exists(), f"No existe: {X_DIR}"
    assert Y_DIR.exists(), f"No existe: {Y_DIR}"

    make_output_dirs()

    pairs = collect_pairs()
    assert len(pairs) > 0, "No se encontraron pares imagen/máscara."

    num_classes, class_ids = explore_dataset(pairs)
    splits = split_and_copy(pairs)

    print_output_structure()

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  Total pares         : {len(pairs)}")
    print(f"  Train               : {len(splits['train'])}")
    print(f"  Validation          : {len(splits['val'])}")
    print(f"  Test                : {len(splits['test'])}")
    print(f"  Clases encontradas  : {num_classes}  →  {class_ids}")
    print(f"  Seed                : {SEED}")
    print("\nListo. Usa NUM_CLASSES =", num_classes, "en el script de entrenamiento.")


if __name__ == "__main__":
    main()
