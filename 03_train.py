"""
03_train.py
===========
Entrenamiento de múltiples modelos de segmentación semántica.
Con una sola ejecución se entrenan todos los experimentos definidos en EXPERIMENTS,
generando carpetas y archivos de resultados independientes para cada uno.

Resultados: output/<ARCHITECTURE>_<BACKBONE>_v<VERSION>/
"""

import time
import json
import random
import traceback
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


# ===========================================================================
# CONFIGURACIÓN GLOBAL — parámetros compartidos por todos los experimentos
# ===========================================================================

# --- Rutas ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"

# --- Versión (se aplica a todos los experimentos) ---
VERSION = 1

# --- Clases ---
# El valor 255 en las máscaras representa el FONDO.
# Las clases de interés son: 0, 1, 2
#
# INCLUDE_BACKGROUND = True  → 255 remapeado a índice 3, se segmenta como cuarta clase
#                               NUM_CLASSES = 4
# INCLUDE_BACKGROUND = False → 255 ignorado (ignore_index), no contribuye a loss ni métricas
#                               NUM_CLASSES = 3
INCLUDE_BACKGROUND = False
IGNORE_INDEX       = 255
NUM_CLASSES        = 4 if INCLUDE_BACKGROUND else 3

# --- Hiperparámetros de entrenamiento ---
EPOCHS      = 2
BATCH_SIZE  = 8
LR          = 1e-4
NUM_WORKERS = 4       # usar 0 en Windows si hay errores de multiprocessing
IMG_SIZE    = 256

# --- Loss ---
# Opciones: "CrossEntropy" | "Dice" | "Jaccard" | "Focal" | "DiceCE"
LOSS_NAME = "DiceCE"

# --- Early stopping ---
EARLY_STOPPING          = True
EARLY_STOPPING_PATIENCE = 10

# --- Data augmentation ---
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP   = True
AUG_ROTATE          = True    # rotación aleatoria ±45°
AUG_COLOR_JITTER    = True    # brillo / contraste / saturación / hue
AUG_RANDOM_CROP     = False
AUG_GAUSSIAN_NOISE  = False

# --- Pesos del encoder ---
ENCODER_WEIGHTS = "imagenet"   # None para inicialización aleatoria

# --- Reproducibilidad ---
SEED = 42

# --- Continuar si un experimento falla ---
# True  → si un experimento lanza una excepción, se registra y se continúa con el siguiente
# False → el script se detiene ante cualquier error
SKIP_ON_ERROR = True

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# EXPERIMENTOS
# Cada entrada: (arquitectura, backbone)
# Arquitecturas: Unet | UnetPlusPlus | FPN | PSPNet | DeepLabV3 | DeepLabV3Plus
#                PAN  | MAnet        | Linknet
# Backbones ligeros : mobilenet_v2, efficientnet-b0
# Backbones medios  : resnet34, resnet50, efficientnet-b3
# Backbones pesados : resnet101, efficientnet-b5, xception
# ===========================================================================
EXPERIMENTS = [
    # (arquitectura,    backbone)           # característica principal
    ("Unet",           "mobilenet_v2"),     # ligero, rápido
    ("Unet",           "resnet34"),         # referencia clásica
    ("UnetPlusPlus",   "resnet50"),         # skip connections densas
    ("FPN",            "resnet50"),         # multi-escala, bueno en objetos pequeños
    ("DeepLabV3Plus",  "resnet101"),        # ASPP, alta capacidad receptiva
    ("MAnet",          "efficientnet-b3"),  # atención multi-escala
]


# ===========================================================================
# UTILIDADES
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy().astype(np.int64)
    if INCLUDE_BACKGROUND:
        out[mask == 255] = 3
    return out


def exp_name(arch: str, backbone: str) -> str:
    return f"{arch}_{backbone}_v{VERSION}"


# ===========================================================================
# DATASET
# ===========================================================================

def build_augmentation(split: str) -> A.Compose:
    transforms = []
    if split == "train":
        if AUG_HORIZONTAL_FLIP:
            transforms.append(A.HorizontalFlip(p=0.5))
        if AUG_VERTICAL_FLIP:
            transforms.append(A.VerticalFlip(p=0.5))
        if AUG_ROTATE:
            transforms.append(A.Rotate(limit=45, p=0.5))
        if AUG_COLOR_JITTER:
            transforms.append(A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
            ))
        if AUG_RANDOM_CROP:
            transforms.append(A.RandomCrop(IMG_SIZE, IMG_SIZE, p=1.0))
        if AUG_GAUSSIAN_NOISE:
            transforms.append(A.GaussNoise(p=0.3))

    transforms += [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


class SegmentationDataset(Dataset):
    def __init__(self, split: str):
        self.img_dir   = DATA_DIR / "images" / split
        self.mask_dir  = DATA_DIR / "masks"  / split
        self.images    = sorted(self.img_dir.glob("*.png"))
        self.transform = build_augmentation(split)
        assert len(self.images) > 0, f"No se encontraron imágenes en {self.img_dir}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = remap_mask(np.array(Image.open(mask_path)))
        aug   = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].long()


# ===========================================================================
# MODELO
# ===========================================================================

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


def build_model(arch: str, backbone: str) -> nn.Module:
    key = arch.lower()
    assert key in ARCH_MAP, f"Arquitectura no reconocida: {arch}"
    return ARCH_MAP[key](
        encoder_name    = backbone,
        encoder_weights = ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = NUM_CLASSES,
    ).to(DEVICE)


# ===========================================================================
# LOSS
# ===========================================================================

def build_loss() -> nn.Module:
    mode = smp.losses.MULTICLASS_MODE
    ig   = -100 if INCLUDE_BACKGROUND else IGNORE_INDEX

    if LOSS_NAME.lower() == "dicece":
        dice = smp.losses.DiceLoss(mode=mode,
                                   ignore_index=ig if not INCLUDE_BACKGROUND else None)
        ce   = nn.CrossEntropyLoss(ignore_index=ig)
        class DiceCELoss(nn.Module):
            def forward(self, pred, target):
                return dice(pred, target) + ce(pred, target)
        return DiceCELoss()

    loss_map = {
        "crossentropy": nn.CrossEntropyLoss(ignore_index=ig),
        "dice":    smp.losses.DiceLoss(mode=mode,
                       ignore_index=ig if not INCLUDE_BACKGROUND else None),
        "jaccard": smp.losses.JaccardLoss(mode=mode,
                       ignore_index=ig if not INCLUDE_BACKGROUND else None),
        "focal":   smp.losses.FocalLoss(mode=mode, ignore_index=ig),
    }
    name = LOSS_NAME.lower()
    assert name in loss_map, \
        f"Loss no reconocida: {LOSS_NAME}. Opciones: CrossEntropy | Dice | Jaccard | Focal | DiceCE"
    return loss_map[name]


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    pred_labels = preds.argmax(dim=1)
    valid = (targets != IGNORE_INDEX) if not INCLUDE_BACKGROUND \
            else torch.ones_like(targets, dtype=torch.bool)

    correct  = ((pred_labels == targets) & valid).sum().item()
    accuracy = correct / (valid.sum().item() + 1e-8)

    iou_per_class, dice_per_class = [], []
    for cls in range(NUM_CLASSES):
        pred_c   = (pred_labels == cls) & valid
        target_c = (targets     == cls) & valid
        tp = (pred_c  &  target_c).sum().item()
        fp = (pred_c  & ~target_c).sum().item()
        fn = (~pred_c &  target_c).sum().item()
        iou_per_class.append(tp / (tp + fp + fn + 1e-8))
        dice_per_class.append((2 * tp) / (2 * tp + fp + fn + 1e-8))

    return {
        "accuracy":       accuracy,
        "mIoU":           float(np.mean(iou_per_class)),
        "iou_per_class":  iou_per_class,
        "mean_dice":      float(np.mean(dice_per_class)),
        "dice_per_class": dice_per_class,
    }


# ===========================================================================
# TRAIN / VAL LOOP
# ===========================================================================

def run_epoch(model, loader, criterion, optimizer, split: str) -> dict:
    is_train = split == "train"
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_targets = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, masks in tqdm(loader, desc=f"  {split}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(masks.detach().cpu())

    metrics = compute_metrics(
        torch.cat(all_preds,   dim=0),
        torch.cat(all_targets, dim=0),
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ===========================================================================
# GUARDADO DE RESULTADOS
# ===========================================================================

def save_curves(history: dict, output_dir: Path, name: str):
    metrics_to_plot = {
        "Loss":     ("train_loss",  "val_loss"),
        "Accuracy": ("train_acc",   "val_acc"),
        "mIoU":     ("train_miou",  "val_miou"),
        "Dice":     ("train_dice",  "val_dice"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Curvas de entrenamiento — {name}", fontsize=13, fontweight="bold")
    for ax, (title, (tk, vk)) in zip(axes.flatten(), metrics_to_plot.items()):
        ep = range(1, len(history[tk]) + 1)
        ax.plot(ep, history[tk], label="Train", linewidth=1.8)
        ax.plot(ep, history[vk], label="Val",   linewidth=1.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"curves_{name}.{ext}", dpi=150,
                    bbox_inches="tight", format=ext)
    plt.close(fig)


def save_metrics_txt(history: dict, best_epoch: int, elapsed: float,
                     output_dir: Path, name: str, arch: str, backbone: str):
    lines = [
        f"Experimento        : {name}",
        f"Arquitectura       : {arch}",
        f"Backbone           : {backbone}",
        f"Épocas totales     : {len(history['train_loss'])}  (best epoch: {best_epoch + 1})",
        f"Tiempo total       : {elapsed / 60:.1f} min",
        f"Device             : {DEVICE}",
        f"Loss fn            : {LOSS_NAME}",
        f"Batch size         : {BATCH_SIZE}",
        f"LR                 : {LR}",
        f"IMG_SIZE           : {IMG_SIZE}",
        f"NUM_CLASSES        : {NUM_CLASSES}",
        f"INCLUDE_BACKGROUND : {INCLUDE_BACKGROUND}  "
        f"(255 → {'clase 3' if INCLUDE_BACKGROUND else 'ignorado'})",
        f"Early stopping     : {EARLY_STOPPING} (patience={EARLY_STOPPING_PATIENCE})",
        "",
        "=" * 50,
        "MEJOR ÉPOCA (val_loss mínimo)",
        "=" * 50,
        f"  val_loss    : {history['val_loss'][best_epoch]:.6f}",
        f"  val_acc     : {history['val_acc'][best_epoch]:.6f}",
        f"  val_mIoU    : {history['val_miou'][best_epoch]:.6f}",
        f"  val_Dice    : {history['val_dice'][best_epoch]:.6f}",
        f"  IoU/clase   : {[round(v,4) for v in history['val_iou_per_class'][best_epoch]]}",
        f"  Dice/clase  : {[round(v,4) for v in history['val_dice_per_class'][best_epoch]]}",
        "",
        "=" * 50,
        "HISTORIAL COMPLETO POR ÉPOCA",
        "=" * 50,
    ]
    header = (f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrmIoU':>7}  {'TrDice':>7}"
              f"  {'VaLoss':>8}  {'VaAcc':>7}  {'VamIoU':>7}  {'VaDice':>7}")
    lines += [header, "-" * len(header)]
    for i in range(len(history["train_loss"])):
        lines.append(
            f"{i+1:>4}  "
            f"{history['train_loss'][i]:>8.5f}  {history['train_acc'][i]:>7.5f}  "
            f"{history['train_miou'][i]:>7.5f}  {history['train_dice'][i]:>7.5f}  "
            f"{history['val_loss'][i]:>8.5f}  {history['val_acc'][i]:>7.5f}  "
            f"{history['val_miou'][i]:>7.5f}  {history['val_dice'][i]:>7.5f}"
        )
    txt_path = output_dir / f"metrics_{name}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Métricas guardadas : {txt_path}")


def save_json(history: dict, best_epoch: int, output_dir: Path,
              name: str, arch: str, backbone: str):
    data = {
        "experiment": name,
        "best_epoch": best_epoch + 1,
        "config": {
            "architecture":       arch,
            "backbone":           backbone,
            "epochs":             EPOCHS,
            "batch_size":         BATCH_SIZE,
            "lr":                 LR,
            "loss":               LOSS_NAME,
            "img_size":           IMG_SIZE,
            "num_classes":        NUM_CLASSES,
            "include_background": INCLUDE_BACKGROUND,
        },
        "history": history,
    }
    json_path = output_dir / f"history_{name}.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Historial JSON     : {json_path}")


# ===========================================================================
# ENTRENAMIENTO DE UN EXPERIMENTO
# ===========================================================================

def train_one(arch: str, backbone: str,
              train_loader: DataLoader, val_loader: DataLoader):
    name       = exp_name(arch, backbone)
    output_dir = PROJECT_ROOT / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experimento : {name}")
    print(f"  Device      : {DEVICE}")
    print(f"{'='*60}")

    set_seed(SEED)
    model     = build_model(arch, backbone)
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [], "train_acc": [], "train_miou": [], "train_dice": [],
        "val_loss":   [], "val_acc":   [], "val_miou":   [], "val_dice":   [],
        "train_iou_per_class": [], "train_dice_per_class": [],
        "val_iou_per_class":   [], "val_dice_per_class":   [],
    }

    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    t_start        = time.time()

    for epoch in range(EPOCHS):
        print(f"\n  Época [{epoch+1}/{EPOCHS}]")
        train_m = run_epoch(model, train_loader, criterion, optimizer, "train")
        val_m   = run_epoch(model, val_loader,   criterion, None,      "val")

        for split, m in (("train", train_m), ("val", val_m)):
            history[f"{split}_loss"].append(m["loss"])
            history[f"{split}_acc"].append(m["accuracy"])
            history[f"{split}_miou"].append(m["mIoU"])
            history[f"{split}_dice"].append(m["mean_dice"])
            history[f"{split}_iou_per_class"].append(m["iou_per_class"])
            history[f"{split}_dice_per_class"].append(m["dice_per_class"])

        print(f"  train → loss:{train_m['loss']:.4f}  acc:{train_m['accuracy']:.4f}"
              f"  mIoU:{train_m['mIoU']:.4f}  dice:{train_m['mean_dice']:.4f}")
        print(f"  val   → loss:{val_m['loss']:.4f}  acc:{val_m['accuracy']:.4f}"
              f"  mIoU:{val_m['mIoU']:.4f}  dice:{val_m['mean_dice']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss  = val_m["loss"]
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(),
                       output_dir / f"best_model_{name}.pth")
            print(f"  >> Mejor modelo guardado (val_loss={best_val_loss:.5f})")
        else:
            patience_count += 1
            print(f"  >> Sin mejora ({patience_count}/{EARLY_STOPPING_PATIENCE})")

        if EARLY_STOPPING and patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping en época {epoch+1}.")
            break

    elapsed = time.time() - t_start
    torch.save(model.state_dict(), output_dir / f"last_model_{name}.pth")

    print(f"\n  Guardando resultados de {name}...")
    save_curves(history, output_dir, name)
    save_metrics_txt(history, best_epoch, elapsed, output_dir, name, arch, backbone)
    save_json(history, best_epoch, output_dir, name, arch, backbone)

    print(f"  Completado en {elapsed/60:.1f} min  |  mejor época: {best_epoch+1}"
          f"  (val_loss={best_val_loss:.5f})")

    # Liberar memoria GPU antes del siguiente experimento
    del model
    torch.cuda.empty_cache()

    return {
        "experiment":  name,
        "best_epoch":  best_epoch + 1,
        "best_val_loss": best_val_loss,
        "elapsed_min": elapsed / 60,
        "status":      "ok",
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  Sesión de entrenamiento — {len(EXPERIMENTS)} experimentos")
    print(f"  Device      : {DEVICE}")
    print(f"  NUM_CLASSES : {NUM_CLASSES}  |  Loss: {LOSS_NAME}"
          f"  |  Epochs: {EPOCHS}  |  BS: {BATCH_SIZE}")
    print(f"{'='*60}")

    # Los dataloaders se crean una sola vez y se reutilizan
    train_ds     = SegmentationDataset("train")
    val_ds       = SegmentationDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"\n  Train: {len(train_ds)} imágenes  |  Val: {len(val_ds)} imágenes")

    summary = []
    session_start = time.time()

    for i, (arch, backbone) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {arch} + {backbone}")
        try:
            result = train_one(arch, backbone, train_loader, val_loader)
        except Exception as e:
            msg = traceback.format_exc()
            print(f"\n  [ERROR] {arch} + {backbone} falló:\n{msg}")
            result = {
                "experiment":    exp_name(arch, backbone),
                "best_epoch":    None,
                "best_val_loss": None,
                "elapsed_min":   None,
                "status":        f"ERROR: {e}",
            }
            if not SKIP_ON_ERROR:
                raise
        summary.append(result)

    # --- Resumen final ---
    total_min = (time.time() - session_start) / 60
    print(f"\n\n{'='*60}")
    print(f"  RESUMEN FINAL  ({total_min:.1f} min totales)")
    print(f"{'='*60}")
    header = f"  {'Experimento':<40}  {'Mejor época':>11}  {'val_loss':>9}  {'min':>6}  Estado"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in summary:
        ep  = str(r["best_epoch"])  if r["best_epoch"]    is not None else "—"
        vl  = f"{r['best_val_loss']:.5f}" if r["best_val_loss"] is not None else "—"
        mn  = f"{r['elapsed_min']:.1f}"   if r["elapsed_min"]   is not None else "—"
        print(f"  {r['experiment']:<40}  {ep:>11}  {vl:>9}  {mn:>6}  {r['status']}")

    # Guardar resumen en output/
    summary_path = PROJECT_ROOT / "output" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
