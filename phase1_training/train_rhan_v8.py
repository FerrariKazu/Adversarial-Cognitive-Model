#!/usr/bin/env python3
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import clip
from autoattack import AutoAttack

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v5 import RHANv5
from model_cornets import CIFARCORnet
from dataset import get_dataloaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_it_features(teacher_model, x):
    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    out = teacher_model.model.module.V1(x_224)
    out = teacher_model.model.module.V2(out)
    out = teacher_model.model.module.V4(out)
    out = teacher_model.model.module.IT(out)
    out = teacher_model.model.module.decoder.avgpool(out)
    out = teacher_model.model.module.decoder.flatten(out)
    return out

def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps, clip_min, clip_max):
    x_natural = x_natural.detach()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()

    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    with torch.no_grad():
        logits_clean = model(x_natural)
        if isinstance(logits_clean, tuple):
            logits_clean = logits_clean[0]
        probs_clean = F.softmax(logits_clean, dim=1).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            if isinstance(logits_adv, tuple):
                logits_adv = logits_adv[0]
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean,
                reduction='batchmean'
            )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    for m in bn_modules:
        m.train()
    return x_adv

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    start_ckpt = os.path.join(ckpt_dir, 'rhan_trades_phase_c_final.pth')
    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v8_best.pth')

    if not os.path.exists(start_ckpt):
        print(f"ERROR: {start_ckpt} not found!")
        return

    # RHANv5 Backbone
    model = RHANv5(head_type='cosine').to(device)
    state = torch.load(start_ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded RHANv5 from {start_ckpt}")

    # CORnet-S teacher
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device, weights_only=False))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Loaded CORnet-S teacher")

    # Frozen CLIP text encoder
    print("Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load('ViT-B/32', device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    PROMPTS = [
        "a photo of an airplane", "a photo of an automobile",
        "a photo of a bird", "a photo of a cat", "a photo of a deer",
        "a photo of a dog", "a photo of a frog", "a photo of a horse",
        "a photo of a ship", "a photo of a truck"
    ]
    with torch.no_grad():
        text_tokens = clip.tokenize(PROMPTS).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)

    trainloader_raw, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=128, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    clip_projector = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    ).to(device)

    epochs = 40
    optimizer = optim.SGD(
        list(model.parameters()) + list(clip_projector.parameters()),
        lr=0.002, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    best_acc = 0.0
    
    eps = 0.031
    beta = 4.0
    steps = 10
    step_size = eps / 4

    print("Starting Training (40 epochs, TRADES eps=0.031)")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = train_correct = total = 0
        l_trades_s = l_clip_s = l_align_s = l_freq_s = 0

        for imgs, lbls in trainloader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            B = imgs.size(0)

            x_adv = generate_trades_adv(model, imgs, step_size, eps, steps, cifar_min, cifar_max)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c, _ = model.forward_with_features(imgs)
                logits_a, feat_adv = model.forward_with_features(x_adv)

                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a, dim=1),
                    F.softmax(logits_c, dim=1),
                    reduction='batchmean'
                )

                feat_projected = F.normalize(clip_projector(feat_adv), dim=-1)
                target_text = text_features[lbls]
                l_clip_anchor = (1 - (feat_projected * target_text).sum(dim=1)).mean()

                with torch.no_grad():
                    it_feats = get_it_features(teacher, x_adv)
                l_align = 1.0 - (F.normalize(feat_adv, dim=-1) * F.normalize(it_feats, dim=-1)).sum(dim=-1).mean()

                if hasattr(model, 'separate_frequencies') and hasattr(model, 'stem_low'):
                    x_low_clean, _ = model.separate_frequencies(imgs)
                    x_low_adv, _ = model.separate_frequencies(x_adv)
                    f_low_clean = model.stem_low(x_low_clean)
                    f_low_adv = model.stem_low(x_low_adv)
                    l_freq = F.mse_loss(f_low_adv, f_low_clean.detach())
                else:
                    l_freq = torch.tensor(0.0, device=device)

                loss = 0.55 * l_trades + 0.20 * l_clip_anchor + 0.15 * l_align + 0.10 * l_freq

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * B
            train_correct += logits_c.argmax(1).eq(lbls).sum().item()
            total += B
            
            l_trades_s += l_trades.item() * B
            l_clip_s += l_clip_anchor.item() * B
            l_align_s += l_align.item() * B
            l_freq_s += l_freq.item() * B

        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(imgs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                _, pred = outputs.max(1)
                test_total += lbls.size(0)
                test_correct += pred.eq(lbls).sum().item()
        test_acc = 100. * test_correct / test_total

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save({
                'model': model.state_dict(),
                'clip_projector': clip_projector.state_dict(),
            }, output_ckpt)
            marker = ' ★'
        else:
            marker = ''

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"L:{train_loss/total:.3f} | TrAcc:{100.*train_correct/total:.1f}% TeAcc:{test_acc:.1f}% | "
              f"T:{l_trades_s/total:.3f} C:{l_clip_s/total:.3f} A:{l_align_s/total:.3f} F:{l_freq_s/total:.3f} | {time.time()-t0:.0f}s{marker}")

    # AutoAttack Evaluation
    print("\nStarting AutoAttack evaluation for automobile (1) and truck (9)...")
    checkpoint = torch.load(output_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
            
    wrapper = Wrapper(model)

    # Collect car and truck images
    car_truck_imgs = []
    car_truck_lbls = []
    for imgs, lbls in testloader:
        mask = (lbls == 1) | (lbls == 9)
        if mask.any():
            car_truck_imgs.append(imgs[mask])
            car_truck_lbls.append(lbls[mask])
    car_truck_imgs = torch.cat(car_truck_imgs)
    car_truck_lbls = torch.cat(car_truck_lbls)

    # Subsample to speed up (e.g. 200 images)
    car_truck_imgs = car_truck_imgs[:200].to(device)
    car_truck_lbls = car_truck_lbls[:200].to(device)
    
    adversary = AutoAttack(wrapper, norm='Linf', eps=0.031, version='custom', attacks_to_run=['apgd-ce'], device=device)
    x_adv_eval = adversary.run_standard_evaluation(car_truck_imgs, car_truck_lbls, bs=100)
    with torch.no_grad():
        outputs = wrapper(x_adv_eval)
        preds = outputs.max(1)[1]
        acc = (preds == car_truck_lbls).float().mean().item() * 100.
        
    print(f"AutoAttack Acc (eps=0.031, cars/trucks): {acc:.2f}%")

if __name__ == '__main__':
    main()
