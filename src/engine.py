# Engine - NAV OCR System

import torch
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(data, device):
    return {k: v.to(device) for k, v in data.items()}


def train_fn(data_loader, model, optimizer, scheduler=None):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader), desc="Trening"):
        data = move_to_device(data, DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        final_loss += loss.item()
    return final_loss / max(len(data_loader), 1)


def eval_fn(data_loader, model):
    model.eval()
    final_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), desc="Evaluering"):
            data = move_to_device(data, DEVICE)
            _, loss = model(**data)
            final_loss += loss.item()
    return final_loss / max(len(data_loader), 1)
