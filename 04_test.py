"""
04_test.py
==========
Evaluación sobre el conjunto de TEST de todos los modelos definidos en EXPERIMENTS.

Para cada experimento:
  - Carga best_model_<name>.pth desde output/<name>/  (si no existe, se omite)
  - Calcula métricas de calidad sobre test y las guarda en TXT y JSON
  - Genera figura comparativa (5 filas x 3 columnas): RGB | máscara ref | predicción
  - Guarda todas las máscaras predichas como PNG con valores de clase originales

Resultados: results/<ARCHITECTURE>_<BACKBONE>_v<VERSION>/
Máscaras  : results/<ARCHITECTURE>_<BACKBONE>_v<VERSION>/masks/

IMPORTANTE: la configuración (clases, tamaño, experimentos, etc.) debe coincidir
con la usada en 03_train.py.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ===========================================================================
# CONFIGURACIÓN — debe coincidir con 03_train.py
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
TRAIN_OUTPUT = PROJECT_ROOT / "output"     # de dónde se leen los modelos
RESULTS_ROOT = PROJECT_ROOT / "results"    # dónde se guardan los resultados de test

VERSION = 1

# --- Clases ---
INCLUDE_BACKGROUND = False
IGNORE_INDEX       = 255
NUM_CLASSES        = 4 if INCLUDE_BACKGROUND else 3

# Mapeo índice de modelo → valor de clase original (para guardar máscaras)
# INCLUDE_BACKGROUND = True : índices 0,1,2,3 → valores 0,1,2,255
# INCLUDE_BACKGROUND = False: índices 0,1,2   → valores 0,1,2
if INCLUDE_BACKGROUND:
    IDX_TO_RAW = {0: 0, 1: 1, 2: 2, 3: 255}
else:
    IDX_TO_RAW = {0: 0, 1: 1, 2: 2}

# --- Inferencia ---
BATCH_SIZE  = 8
NUM_WORKERS = 4         # 0 en Windows si hay errores de multiprocessing
IMG_SIZE    = 256

SEED = 42

# --- Figura comparativa ---
N_FIG_SAMPLES = 5       # filas de la figura

# Mapa de colores para la VISUALIZACIÓN (no afecta las máscaras PNG guardadas)
CLASS_COLOR_MAP = {
    0:   (0,   0,   0),    # negro
    1:   (34,  139, 34),   # verde
    2:   (30,  144, 255),  # azul
    255: (255, 255, 255),  # blanco (fondo)
}
CLASS_LABELS = {
    0:   "Clase 0",
    1:   "Clase 1",
    2:   "Clase 2",
    255: "Fondo (255)",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Experimentos (misma lista que en 03_train.py) ---
EXPERIMENTS = [
    ("Unet",           "mobilenet_v2"),
    ("Unet",           "resnet34"),
    ("UnetPlusPlus",   "resnet50"),
    ("FPN",            "resnet50"),
    ("DeepLabV3Plus",  "resnet101"),
    ("MAnet",          "efficientnet-b3"),
]

ARCH_MAP = {
    "unet":          smp.Unet,
    "unetplusplus":  smp.UnetPlusPlus,
    "fpn":           smp.FPN,
    "pspnet":        smp.PSPNet,
    "deeplabv3":     smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "pan":           smp.PAN,
    "manet":         smp.MAnet,
    "linknet":       smp.Linknet,
}


# ===========================================================================
# UTILIDADES
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exp_name(arch: str, backbone: str) -> str:
    return f"{arch}_{backbone}_v{VERSION}"


def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy().astype(np.int64)
    if INCLUDE_BACKGROUND:
        out[mask == 255] = 3
    return out


def pred_to_raw(pred_idx: np.ndarray) -> np.ndarray:
    """Convierte índices del modelo a valores de clase originales del dataset."""
    out = np.zeros_like(pred_idx, dtype=np.uint8)
    for idx, raw in IDX_TO_RAW.items():
        out[pred_idx == idx] = raw
    return out


def colorize(raw_mask: np.ndarray) -> np.ndarray:
    """Colorea una máscara con valores originales (0,1,2,255) para visualización."""
    h, w = raw_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in CLASS_COLOR_MAP.items():
        rgb[raw_mask == val] = color
    return rgb


# ===========================================================================
# DATASET DE TEST
# ===========================================================================

# Normalización idéntica a entrenamiento
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

TEST_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])


class TestDataset(Dataset):
    def __init__(self):
        self.img_dir  = DATA_DIR / "images" / "test"
        self.mask_dir = DATA_DIR / "masks"  / "test"
        self.images   = sorted(self.img_dir.glob("*.png"))
        assert len(self.images) > 0, f"No se encontraron imágenes en {self.img_dir}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        image_raw = np.array(Image.open(img_path).convert("RGB"))
        mask_raw  = np.array(Image.open(mask_path))           # valores originales
        mask_idx  = remap_mask(mask_raw)                      # índices del modelo

        aug = TEST_TRANSFORM(image=image_raw, mask=mask_idx)
        image  = aug["image"]
        target = aug["mask"].long()

        # Para visualización guardamos también la imagen redimensionada sin normalizar
        img_vis  = A.Resize(IMG_SIZE, IMG_SIZE)(image=image_raw)["image"]
        mask_vis = A.Resize(IMG_SIZE, IMG_SIZE,
                            interpolation=0)(image=mask_raw)["image"]  # nearest

        return {
            "image":     image,
            "target":    target,
            "img_vis":   img_vis,
            "mask_vis":  mask_vis.astype(np.int64),
            "filename":  img_path.name,
        }


def collate(batch):
    return {
        "image":    torch.stack([b["image"]  for b in batch]),
        "target":   torch.stack([b["target"] for b in batch]),
        "img_vis":  [b["img_vis"]  for b in batch],
        "mask_vis": [b["mask_vis"] for b in batch],
        "filename": [b["filename"] for b in batch],
    }


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def accumulate_confusion(pred_labels, targets, conf):
    """Acumula matriz de confusión (NUM_CLASSES x NUM_CLASSES) sobre píxeles válidos."""
    if not INCLUDE_BACKGROUND:
        valid = (targets != IGNORE_INDEX)
    else:
        valid = torch.ones_like(targets, dtype=torch.bool)

    p = pred_labels[valid].view(-1)
    t = targets[valid].view(-1)
    k = (t * NUM_CLASSES + p)
    binc = torch.bincount(k, minlength=NUM_CLASSES**2)
    conf += binc.reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def metrics_from_confusion(conf: torch.Tensor) -> dict:
    """Calcula métricas de calidad a partir de la matriz de confusión acumulada."""
    conf = conf.double()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    iou_per_class  = (tp / (tp + fp + fn + 1e-8)).tolist()
    dice_per_class = (2 * tp / (2 * tp + fp + fn + 1e-8)).tolist()
    prec_per_class = (tp / (tp + fp + 1e-8)).tolist()
    rec_per_class  = (tp / (tp + fn + 1e-8)).tolist()

    pixel_acc = (tp.sum() / (conf.sum() + 1e-8)).item()

    return {
        "pixel_accuracy":     pixel_acc,
        "mIoU":               float(np.mean(iou_per_class)),
        "mean_dice":          float(np.mean(dice_per_class)),
        "mean_precision":     float(np.mean(prec_per_class)),
        "mean_recall":        float(np.mean(rec_per_class)),
        "iou_per_class":      iou_per_class,
        "dice_per_class":     dice_per_class,
        "precision_per_class": prec_per_class,
        "recall_per_class":   rec_per_class,
    }


# ===========================================================================
# MODELO
# ===========================================================================

def build_model(arch: str, backbone: str) -> nn.Module:
    key = arch.lower()
    assert key in ARCH_MAP, f"Arquitectura no reconocida: {arch}"
    # encoder_weights=None: los pesos se cargan desde el checkpoint entrenado
    return ARCH_MAP[key](
        encoder_name    = backbone,
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
    ).to(DEVICE)


# ===========================================================================
# GUARDADO
# ===========================================================================

def save_metrics(metrics: dict, conf: torch.Tensor, output_dir: Path, name: str,
                 arch: str, backbone: str, n_images: int):
    # --- TXT ---
    raw_vals = list(IDX_TO_RAW.values())
    conf_np  = conf.cpu().numpy().astype(np.int64)
    lines = [
        f"Evaluación TEST    : {name}",
        f"Arquitectura       : {arch}",
        f"Backbone           : {backbone}",
        f"Imágenes de test   : {n_images}",
        f"NUM_CLASSES        : {NUM_CLASSES}",
        f"INCLUDE_BACKGROUND : {INCLUDE_BACKGROUND}",
        f"Valores de clase   : {raw_vals}",
        f"Device             : {DEVICE}",
        "",
        "=" * 50,
        "MÉTRICAS GLOBALES",
        "=" * 50,
        f"  Pixel Accuracy  : {metrics['pixel_accuracy']:.6f}",
        f"  mIoU            : {metrics['mIoU']:.6f}",
        f"  Mean Dice (F1)  : {metrics['mean_dice']:.6f}",
        f"  Mean Precision  : {metrics['mean_precision']:.6f}",
        f"  Mean Recall     : {metrics['mean_recall']:.6f}",
        "",
        "=" * 50,
        "MÉTRICAS POR CLASE",
        "=" * 50,
        f"  {'Clase':>10}  {'IoU':>8}  {'Dice':>8}  {'Prec':>8}  {'Recall':>8}",
        "  " + "-" * 50,
    ]
    for i, raw in enumerate(raw_vals):
        lines.append(
            f"  {('val '+str(raw)):>10}  "
            f"{metrics['iou_per_class'][i]:>8.4f}  "
            f"{metrics['dice_per_class'][i]:>8.4f}  "
            f"{metrics['precision_per_class'][i]:>8.4f}  "
            f"{metrics['recall_per_class'][i]:>8.4f}"
        )

    # Matriz de confusión (conteos absolutos de píxeles): filas = referencia, columnas = predicción
    lines += [
        "",
        "=" * 50,
        "MATRIZ DE CONFUSIÓN (píxeles)",
        "filas = clase real (referencia)  |  columnas = clase predicha",
        "=" * 50,
    ]
    col_header = "  " + " " * 12 + "".join(f"{('pred '+str(r)):>14}" for r in raw_vals)
    lines.append(col_header)
    for i, raw in enumerate(raw_vals):
        row = "".join(f"{conf_np[i, j]:>14d}" for j in range(NUM_CLASSES))
        lines.append(f"  {('real '+str(raw)):>12}{row}")

    txt_path = output_dir / f"test_metrics_{name}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Métricas TXT  : {txt_path}")

    # --- JSON ---
    data = {
        "experiment": name,
        "architecture": arch,
        "backbone": backbone,
        "n_test_images": n_images,
        "num_classes": NUM_CLASSES,
        "include_background": INCLUDE_BACKGROUND,
        "class_values": raw_vals,
        "metrics": metrics,
        "confusion_matrix": conf_np.tolist(),
        "confusion_matrix_axes": "rows=real, cols=pred",
    }
    json_path = output_dir / f"test_metrics_{name}.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Métricas JSON : {json_path}")


def save_confusion_figure(conf: torch.Tensor, output_dir: Path, name: str):
    """
    Genera figura de la matriz de confusión en dos paneles:
      izquierda  → conteos absolutos
      derecha    → normalizada por fila (recall por clase)
    Filas = clase real, columnas = clase predicha.
    """
    conf_np = conf.cpu().numpy().astype(np.float64)
    row_sums = conf_np.sum(axis=1, keepdims=True)
    conf_norm = conf_np / (row_sums + 1e-8)

    raw_vals = list(IDX_TO_RAW.values())
    labels   = [str(r) for r in raw_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(f"Matriz de confusión — {name}", fontsize=13, fontweight="bold")

    panels = [
        (axes[0], conf_np,   "Conteos absolutos", "d",   "Blues"),
        (axes[1], conf_norm, "Normalizada por fila (recall)", ".2f", "Greens"),
    ]
    for ax, mat, title, fmt, cmap in panels:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Referencia (real)")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(labels)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(labels)
        # Anotar cada celda
        thresh = mat.max() / 2.0 if mat.max() > 0 else 0.5
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = mat[i, j]
                txt = f"{int(val):d}" if fmt == "d" else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"test_confusion_{name}.{ext}", dpi=150,
                    bbox_inches="tight", format=ext)
    plt.close(fig)
    print(f"  Matriz conf.  : test_confusion_{name}.png / .svg")


def save_comparison_figure(samples: list, output_dir: Path, name: str):
    """
    samples: lista de dicts con keys: img_vis, mask_ref_raw, pred_raw, filename
    Figura de N_FIG_SAMPLES filas x 3 columnas: RGB | máscara ref | predicción
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2), constrained_layout=True)
    fig.suptitle(f"Resultados TEST — {name}", fontsize=14, fontweight="bold")

    if n == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["Imagen RGB", "Máscara referencia", "Predicción"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=11, fontweight="bold")

    for row, s in enumerate(samples):
        axes[row, 0].imshow(s["img_vis"])
        axes[row, 0].set_ylabel(s["filename"], fontsize=7, rotation=0,
                                ha="right", va="center", labelpad=35)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        axes[row, 1].imshow(colorize(s["mask_ref_raw"]))
        axes[row, 1].axis("off")

        axes[row, 2].imshow(colorize(s["pred_raw"]))
        axes[row, 2].axis("off")

    legend_patches = [
        mpatches.Patch(color=tuple(c/255 for c in CLASS_COLOR_MAP[v]),
                       label=CLASS_LABELS[v])
        for v in CLASS_COLOR_MAP
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(CLASS_COLOR_MAP), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"test_comparison_{name}.{ext}", dpi=150,
                    bbox_inches="tight", format=ext)
    plt.close(fig)
    print(f"  Figura comp.  : test_comparison_{name}.png / .svg")


# ===========================================================================
# EVALUACIÓN DE UN MODELO
# ===========================================================================

def evaluate_one(arch: str, backbone: str, loader: DataLoader, n_images: int):
    name        = exp_name(arch, backbone)
    model_path  = TRAIN_OUTPUT / name / f"best_model_{name}.pth"

    if not model_path.exists():
        print(f"  [OMITIDO] No existe el modelo: {model_path}")
        return {"experiment": name, "status": "modelo_no_encontrado"}

    results_dir = RESULTS_ROOT / name
    masks_dir   = results_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Cargando modelo: {model_path}")
    model = build_model(arch, backbone)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    fig_samples = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {name}", leave=False):
            images  = batch["image"].to(DEVICE)
            targets = batch["target"]                       # CPU, índices
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu()           # (B,H,W) índices

            conf = accumulate_confusion(preds, targets, conf)

            # Guardar cada máscara predicha como PNG (valores originales)
            for i in range(preds.size(0)):
                pred_raw = pred_to_raw(preds[i].numpy())
                Image.fromarray(pred_raw, mode="L").save(
                    masks_dir / batch["filename"][i]
                )

                # Recolectar muestras para la figura comparativa
                if len(fig_samples) < N_FIG_SAMPLES:
                    fig_samples.append({
                        "img_vis":      batch["img_vis"][i],
                        "mask_ref_raw": np.array(
                            Image.open(DATA_DIR / "masks" / "test" / batch["filename"][i])
                        ),
                        "pred_raw":     pred_raw,
                        "filename":     batch["filename"][i],
                    })

    metrics = metrics_from_confusion(conf)

    print(f"  Guardando resultados de {name}...")
    save_metrics(metrics, conf, results_dir, name, arch, backbone, n_images)
    save_confusion_figure(conf, results_dir, name)
    save_comparison_figure(fig_samples, results_dir, name)
    print(f"  Máscaras PNG  : {masks_dir}  ({n_images} archivos)")

    del model
    torch.cuda.empty_cache()

    return {
        "experiment":     name,
        "status":         "ok",
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mIoU":           metrics["mIoU"],
        "mean_dice":      metrics["mean_dice"],
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    set_seed(SEED)
    print(f"\n{'='*60}")
    print(f"  Evaluación TEST — {len(EXPERIMENTS)} modelos")
    print(f"  Device      : {DEVICE}")
    print(f"  NUM_CLASSES : {NUM_CLASSES}  |  INCLUDE_BACKGROUND: {INCLUDE_BACKGROUND}")
    print(f"{'='*60}")

    test_ds = TestDataset()
    # shuffle=False para que las muestras de la figura sean reproducibles
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             collate_fn=collate)
    n_images = len(test_ds)
    print(f"\n  Imágenes de test: {n_images}")

    summary = []
    for i, (arch, backbone) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {arch} + {backbone}")
        try:
            result = evaluate_one(arch, backbone, test_loader, n_images)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {arch}+{backbone}:\n{traceback.format_exc()}")
            result = {"experiment": exp_name(arch, backbone), "status": f"ERROR: {e}"}
        summary.append(result)

    # --- Resumen comparativo de todos los modelos ---
    print(f"\n\n{'='*60}")
    print("  RESUMEN COMPARATIVO — TEST")
    print(f"{'='*60}")
    header = f"  {'Experimento':<40}  {'PixAcc':>8}  {'mIoU':>8}  {'Dice':>8}  Estado"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in summary:
        if r.get("status") == "ok":
            print(f"  {r['experiment']:<40}  "
                  f"{r['pixel_accuracy']:>8.4f}  {r['mIoU']:>8.4f}  "
                  f"{r['mean_dice']:>8.4f}  ok")
        else:
            print(f"  {r['experiment']:<40}  {'—':>8}  {'—':>8}  {'—':>8}  {r['status']}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_ROOT / "test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
