import torch
import torch.nn.functional as F


def pgd_attack(
    model,
    image_tensor,
    label_idx,
    eps: float = 0.008,
    steps: int = 40,
    step_size: float = None,
    random_start: bool = True,
    domain_min: float = 0.0,
    domain_max: float = 1.0,
):
    step_size = step_size or eps / steps * 2.0
    x_adv = image_tensor.clone().detach()
    if random_start:
        x_adv += torch.zeros_like(x_adv).uniform_(-eps, eps)
    with torch.no_grad():
        x_adv = torch.clamp(x_adv, domain_min, domain_max)
    x_adv.requires_grad_(True)

    for _ in range(steps):
        logits = model(x_adv.unsqueeze(0))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss = F.cross_entropy(logits, torch.tensor([label_idx], device=x_adv.device))
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - image_tensor, -eps, eps)
            x_adv = torch.clamp(image_tensor + delta, domain_min, domain_max)
        x_adv.requires_grad_(True)

    return x_adv.detach()


def compute_accuracy(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                output = model(images)
                if isinstance(output, (tuple, list)):
                    output = output[0]
            correct += output.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)
