from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def knowledge_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    valid: Tensor | None = None,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    selected = (
        torch.ones(student_logits.shape[0], dtype=torch.bool, device=student_logits.device)
        if valid is None
        else valid.to(student_logits.device).bool()
    )
    if selected.ndim != 1 or selected.shape[0] != student_logits.shape[0]:
        raise ValueError("distillation validity mask must have shape [B]")
    if not bool(selected.any()):
        return student_logits.sum() * 0
    student = student_logits[selected]
    teacher = teacher_logits.detach().to(student.device)[selected]
    scale = float(temperature)
    if student.ndim == 1 or student.shape[-1] == 1:
        student = student.reshape(-1)
        teacher = teacher.reshape(-1)
        soft_target = torch.sigmoid(teacher / scale)
        return functional.binary_cross_entropy_with_logits(student / scale, soft_target) * scale**2
    teacher_probability = functional.softmax(teacher / scale, dim=-1)
    student_probability = functional.log_softmax(student / scale, dim=-1)
    return functional.kl_div(student_probability, teacher_probability, reduction="batchmean") * scale**2


def distillation_loss(
    supervised_loss: Tensor,
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    alpha_supervised: float = 1.0,
    alpha_distill: float = 1.0,
    valid: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if alpha_supervised < 0 or alpha_distill < 0 or alpha_supervised + alpha_distill <= 0:
        raise ValueError("distillation weights must be non-negative with a positive sum")
    knowledge = knowledge_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        valid=valid,
    )
    total = float(alpha_supervised) * supervised_loss + float(alpha_distill) * knowledge
    return total, {"supervised": supervised_loss.detach(), "distill": knowledge.detach()}
