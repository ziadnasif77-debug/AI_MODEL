# Main - NAV OCR System

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from trainer import ModelModule
from loader import dataSet
from engine import train_fn, eval_fn, DEVICE
from config import (
    MODEL_DIR, TRAINING_JSON, OUTPUT_MODEL_DIR,
    NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VALIDATION_SPLIT, SAVE_EVERY, EARLY_STOPPING_PATIENCE,
    DATALOADER_WORKERS, RANDOM_SEED
)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

featur_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokeniser = LayoutLMv3TokenizerFast.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
processor = LayoutLMv3Processor(tokenizer=tokeniser, feature_extractor=featur_extractor)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("NAV OCR - LayoutLMv3 Treningsprogram")
    print(f"Enhet: {DEVICE}")
    print("=" * 60)

    # Last inn datasett
    full_ds = dataSet(TRAINING_JSON, processor)
    total = len(full_ds)
    val_size = max(1, int(total * VALIDATION_SPLIT))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

    print(f"Totalt: {total} bilder | Trening: {train_size} | Validering: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=DATALOADER_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=DATALOADER_WORKERS, pin_memory=True
    )

    # Opprett modell og flytt til GPU/CPU
    model = ModelModule(NUM_CLASSES).to(DEVICE)

    # Optimizer og scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\nStarter trening: {EPOCHS} epoker")
    print(f"Early stopping: {EARLY_STOPPING_PATIENCE} epoker uten forbedring")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # Trening
        train_loss = train_fn(train_loader, model, optimizer, scheduler)
        train_losses.append(train_loss)

        # Validering
        val_loss = eval_fn(val_loader, model)
        val_losses.append(val_loss)

        print(f"  Epoke {epoch+1:2d}/{EPOCHS}  |  Trening: {train_loss:.4f}  |  Validering: {val_loss:.4f}")

        # Lagre beste modell basert på validerings-loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, 'model_best.bin'))
            print(f"  -> Ny beste modell! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Lagre sjekkpunkt
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, f'model_epoke_{epoch+1}.bin'))
            print(f"  -> Sjekkpunkt lagret: epoke {epoch+1}")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Tidlig stopp! Ingen forbedring i {EARLY_STOPPING_PATIENCE} epoker")
            break

    # Lagre loss-historikk
    np.savez(
        os.path.join(OUTPUT_MODEL_DIR, 'loss_history.npz'),
        train=np.array(train_losses),
        val=np.array(val_losses)
    )

    print("=" * 60)
    print("Trening fullfort!")
    print(f"  Epoker kjort:    {len(train_losses)}/{EPOCHS}")
    print(f"  Beste val loss:  {best_val_loss:.4f}")
    print(f"  Modell lagret i: {OUTPUT_MODEL_DIR}")
    print("=" * 60)
