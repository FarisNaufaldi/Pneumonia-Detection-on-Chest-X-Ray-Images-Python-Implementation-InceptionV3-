# train_inception_timm_chestxray.py
# pip install timm torchvision torch tqdm  (di Kaggle biasanya sdh ada)
import os, math, time, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import timm
from timm.data import resolve_data_config, create_transform

# =========================
# CONFIG
# =========================
MODEL_ID       = "inception_v3"   # bisa ganti model timm lain
BATCH_SIZE     = 8
EPOCHS_PHASE1  = 5
EPOCHS_PHASE2  = 10
LR_PHASE1      = 5e-5
LR_PHASE2      = 2e-5
WEIGHT_DECAY   = 3e-3
SEED           = 42

# Kaggle dataset path (per screenshot)
DATA_DIR       = "/kaggle/input/chest-xray-pneumonia/chest_xray"
MODEL_DIR      = "./Model"
MODEL_PATH     = "./Model/best_inception_v3_timm.pth"  # hanya untuk SAVE best

UNFREEZE_SUBSTR = ["Mixed_7", "Mixed_6e"]  # blok yang dibuka saat fine-tune
USE_AMP         = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# UTILS
# =========================
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def pick_split(root: Path, names) -> Path | None:
    for n in names:
        p = root / n
        if p.exists() and any(p.iterdir()):
            return p
    return None

def compute_class_weights(samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    counts = [0]*num_classes
    for _, y in samples: counts[y] += 1
    total = sum(counts); k = float(num_classes)
    w = [(total/(k*c)) if c>0 else 0.0 for c in counts]
    if num_classes == 2:
        w[0] *= 1.3; w[1] *= 1.1
    return torch.tensor(w, dtype=torch.float32, device=device)

class RandomRotateBy90:
    def __call__(self, img: Image.Image):
        k = random.choice([0, 1, 2, 3])
        if k == 0:  return img
        if k == 1:  return img.transpose(Image.ROTATE_90)
        if k == 2:  return img.transpose(Image.ROTATE_180)
        return img.transpose(Image.ROTATE_270)

@torch.no_grad()
def eval_cls(model: nn.Module, loader: DataLoader):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss=0.0; tot_correct=0; tot_seen=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30,30)
        loss = crit(logits, y)
        tot_loss += loss.item()
        pred = logits.argmax(1)
        tot_correct += (pred==y).sum().item()
        tot_seen += y.size(0)
    return tot_loss/max(1,len(loader)), tot_correct/max(1,tot_seen)

@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss=0.0; tot_correct=0; tot_seen=0
    tp=tn=fp=fnn=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30,30)
        loss = crit(logits, y)
        tot_loss += loss.item()
        pred = logits.argmax(1)
        tot_correct += (pred==y).sum().item()
        tot_seen += y.size(0)
        pos=(y==1); neg=(y==0); pp=(pred==1); pn=(pred==0)
        tp += (pp&pos).sum().item(); tn += (pn&neg).sum().item()
        fp += (pp&neg).sum().item(); fnn+= (pn&pos).sum().item()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fnn) if (tp+fnn)>0 else 0.0
    beta2 = 4.0
    f2 = (1+beta2)*(prec*rec)/(beta2*prec+rec) if (prec+rec)>0 else 0.0
    return tot_loss/max(1,len(loader)), tot_correct/max(1,tot_seen), prec, rec, f2

def freeze_all_except_classifier(model: nn.Module):
    for n,p in model.named_parameters():
        p.requires_grad = False
    # aktifkan head/classifier (nama umum di timm)
    head_keys = ("classifier","fc","head","last_linear")
    for n,p in model.named_parameters():
        if any(hk in n for hk in head_keys):
            p.requires_grad = True

def unfreeze_by_substrings(model: nn.Module, substrs):
    freeze_all_except_classifier(model)
    for n,p in model.named_parameters():
        if any(s in n for s in substrs):
            p.requires_grad = True

def train_phase(model, train_loader, valid_loader, lr, epochs, phase_name, class_weights, save_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best_acc = 0.0

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(total=len(train_loader.dataset), unit="img", desc=f"[{phase_name} {ep}/{epochs}]")
        run_loss=0.0; run_correct=0; seen=0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30,30)
                loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            run_loss += loss.item()
            pred = logits.argmax(1)
            run_correct += (pred==y).sum().item()
            seen += y.size(0)

            avg_loss = run_loss / max(1, math.ceil(seen/BATCH_SIZE))
            avg_acc  = 100.0*run_correct/max(1,seen)
            pbar.set_postfix_str(f"loss {avg_loss:.4f} • acc {avg_acc:.2f}%")
            pbar.update(y.size(0))
        pbar.close()

        val_loss, val_acc = eval_cls(model, valid_loader)
        print(f"→ val_loss {val_loss:.4f} • val_acc {val_acc*100:.2f}%")
        if not math.isnan(val_acc) and val_acc > best_acc:
            best_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (val acc: {best_acc*100:.2f}%)")
    return best_acc

# =========================
# MAIN
# =========================
def main():
    t0 = time.time(); set_seed(SEED)
    print(f"Device: {device} | CPU threads: {os.cpu_count()}")

    root = Path(DATA_DIR)
    train_dir = root / "train"
    test_dir  = root / "test"
    valid_dir = pick_split(root, ["val", "valid"]) or test_dir
    if valid_dir is test_dir:
        print("⚠ 'val/valid' tidak ada → pakai TEST sebagai validation.\n")

    # classes
    tmp = ImageFolder(train_dir)
    classes = tmp.classes; num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # model (load dari timm; fallback ke random init kalau gagal download weights)
    try:
        model = timm.create_model(MODEL_ID, pretrained=True, num_classes=num_classes).to(device)
    except Exception as e:
        print(f"⚠ Tidak bisa load pretrained timm ({e}) → pakai random init.")
        model = timm.create_model(MODEL_ID, pretrained=False, num_classes=num_classes).to(device)

    # transforms (auto sesuai model); tambah augment ringan biar mirip versi Rust
    cfg = resolve_data_config({}, model=model)
    tf_train = create_transform(**cfg, is_training=True)
    tf_eval  = create_transform(**cfg, is_training=False)

    # Prepend extra augments (Rotate90 + VerticalFlip)
    tf_train = T.Compose([RandomRotateBy90(), T.RandomVerticalFlip(p=0.5)] + list(tf_train.transforms))

    # datasets & loaders
    ds_train = ImageFolder(train_dir, transform=tf_train)
    ds_valid = ImageFolder(valid_dir, transform=tf_eval)
    ds_test  = ImageFolder(test_dir,  transform=tf_eval)

    class_weights = compute_class_weights(ds_train.samples, num_classes)
    print(f"Class weights: {class_weights.tolist()}")

    num_workers = min(4, max(0, (os.cpu_count() or 2) - 1))
    pin = torch.cuda.is_available()
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # ===== PHASE 1: Freeze backbone, train classifier =====
    print("\n===== PHASE 1: Frozen base (classifier only) =====")
    freeze_all_except_classifier(model)
    _ = train_phase(model, dl_train, dl_valid, LR_PHASE1, EPOCHS_PHASE1, "Phase-1", class_weights, MODEL_PATH)

    # load best dari phase-1 (kalau ada)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # ===== PHASE 2: Fine-tune sebagian blok =====
    print("\n===== PHASE 2: Fine-tuning (unfreeze partial base) =====")
    unfreeze_by_substrings(model, UNFREEZE_SUBSTR)
    _ = train_phase(model, dl_train, dl_valid, LR_PHASE2, EPOCHS_PHASE2, "Phase-2", class_weights, MODEL_PATH)

    # pakai best sebelum test
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # ===== TEST =====
    print("\nEvaluating on test set...")
    if num_classes == 2:
        _loss, acc, prec, rec, f2 = eval_binary(model, dl_test)
        print(f"✓ Test Acc:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F2-score:  {f2*100:.2f}%")
    else:
        loss, acc = eval_cls(model, dl_test)
        print(f"✓ Test Loss: {loss:.4f} | Test Accuracy: {acc*100:.2f}%")

    print(f"\n⏱ Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()

