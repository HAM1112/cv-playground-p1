"""Train CellNet on the synthetic (+ optional real) cell crops."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import DEFAULT_CACHE, build_synthetic_cells, load_cells, load_real_cells, save_cells
from .model import DEFAULT_MODEL_PATH, CellNet, pick_device, preprocess, save_model


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Light on-the-fly augmentation on a (B, 1, H, W) batch in [0, 1]."""
    B, _, H, W = x.shape
    # Random shift up to 4 px via padding + crop.
    pad = 4
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (B,), generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (B,), generator=gen)
    out = torch.empty_like(x)
    for i in range(B):
        out[i] = xp[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
    x = out
    # Flips (circles and empty cells are symmetric).
    flip_h = torch.rand(B, generator=gen) < 0.5
    flip_v = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip_h[:, None, None, None], x.flip(-1), x)
    x = torch.where(flip_v[:, None, None, None], x.flip(-2), x)
    # Brightness / contrast jitter + noise.
    gain = torch.empty(B, 1, 1, 1).uniform_(0.7, 1.3, generator=gen)
    bias = torch.empty(B, 1, 1, 1).uniform_(-0.15, 0.15, generator=gen)
    noise = torch.randn(x.shape, generator=gen) * 0.03
    return ((x - 0.5) * gain + 0.5 + bias + noise).clamp_(0, 1)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, bs: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs].to(device))
        correct += (logits.argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / max(len(X), 1)


def train(
    data_path: Path | str = DEFAULT_CACHE,
    out_path: Path | str = DEFAULT_MODEL_PATH,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    n_cells_if_missing: int = 20000,
    real_share: float = 0.15,
    photos_dir: Path | str | None = None,
) -> dict:
    from .progress import Progress, fmt_secs, stage

    data_path = Path(data_path)
    n_stages = 3
    t_start = time.time()

    # ---------------------------------------------------------------- 1. photos
    stage(1, n_stages, "harvesting cell crops from your photos")
    if photos_dir is not None:
        from .harvest import harvest_labeled

        stats = harvest_labeled(photos_dir, verbose=True)
        n_photo_crops = 0
        for folder, st in stats.items():
            if st["photos"]:
                n_photo_crops += st["circle"] + st["empty"]
                print(f"  photos/{folder}: {st['photos']} photos -> {st['circle']} circle, {st['empty']} empty crops")
                if st["no_boxes"]:
                    print(f"  no boxes found in: {', '.join(st['no_boxes'])}")
                if st["suspicious"]:
                    print(f"  WARNING every box looks filled (should these be in full/?): {', '.join(st['suspicious'])}")
        if n_photo_crops == 0:
            print(f"  no photos found under {photos_dir} (put some in full/ and available/ to use them)")
    else:
        print("  skipped (--no-photos)")

    # ---------------------------------------------------------------- 2. synthetic data
    stage(2, n_stages, "synthetic training data")
    if not data_path.exists():
        print(f"  no cached dataset at {data_path.name}; generating {n_cells_if_missing} synthetic cells (one-off, a few minutes)")
        X, y, stats = build_synthetic_cells(n_cells_if_missing, seed=seed)
        save_cells(data_path, X, y)
        print(f"  saved to {data_path}  (detect rate {100 * stats['detect_rate']:.1f}%, circle fraction {stats['circle_frac']:.2f})")
    else:
        print(f"  using cached {data_path.name}")
    X_np, y_np = load_cells(data_path)
    print(f"  {len(y_np)} synthetic cells loaded")
    Xr, yr = load_real_cells()
    if len(yr):
        # A few hundred real crops would vanish next to 20k synthetic ones, so repeat
        # them until they make up about `real_share` of the training data.
        repeat = max(1, int(round(real_share * len(y_np) / ((1 - real_share) * len(yr)))))
        print(f"  {len(yr)} real crops repeated x{repeat} so they are ~{100 * real_share:.0f}% of the data")
        X_np = np.concatenate([X_np] + [Xr] * repeat)
        y_np = np.concatenate([y_np] + [yr] * repeat)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_np))
    n_val = max(1, int(0.2 * len(perm)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X = preprocess(X_np)
    y = torch.from_numpy(y_np).long()
    Xtr, ytr, Xval, yval = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    # ---------------------------------------------------------------- 3. training
    stage(3, n_stages, "training CellNet")
    dev = pick_device(device)
    model = CellNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    n_batches = (len(ytr) + batch_size - 1) // batch_size
    print(f"  device {dev} | {len(ytr)} train / {len(yval)} validation cells | "
          f"{epochs} epochs x {n_batches} batches of {batch_size} | {sum(p.numel() for p in model.parameters())} parameters")

    best_acc, best_state, best_epoch = 0.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(ytr), generator=gen)
        total_loss = 0.0
        seen = 0
        bar = Progress(n_batches, f"  epoch {epoch:2d}/{epochs}")
        for b, i in enumerate(range(0, len(order), batch_size), start=1):
            idx = order[i:i + batch_size]
            xb = _augment(Xtr[idx], gen).to(dev)
            yb = ytr[idx].to(dev)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            seen += len(idx)
            bar.update(b, extra=f"loss {total_loss / seen:.4f}")
        sched.step()
        acc = evaluate(model, Xval, yval, dev)
        improved = acc >= best_acc
        if improved:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elapsed = time.time() - t0
        eta = elapsed / epoch * (epochs - epoch)
        bar.close(
            f"  epoch {epoch:2d}/{epochs}  loss {total_loss / len(ytr):.4f}  "
            f"val acc {100 * acc:6.2f}%{'  *best*' if improved else ''}  "
            f"[{fmt_secs(elapsed)} elapsed, ~{fmt_secs(eta)} left]"
        )

    model.load_state_dict(best_state)
    save_model(model.cpu(), out_path)
    print(f"\nDone in {fmt_secs(time.time() - t_start)}. Best validation accuracy {100 * best_acc:.2f}% "
          f"(epoch {best_epoch}) saved to {out_path}")
    return {"val_acc": best_acc, "n_train": len(ytr), "n_val": len(yval), "device": str(dev)}
