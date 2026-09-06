from __future__ import annotations

import copy

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .base import DAPTObjective


class DINODAPT(DAPTObjective):
    def __init__(self, encoder, projection_dim: int = 128, teacher_momentum: float = 0.996):
        super().__init__(encoder)
        self.student_head = nn.Linear(encoder.feature_dim, projection_dim)
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.teacher_momentum = teacher_momentum
        for module in (self.teacher_encoder, self.teacher_head):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def update_teacher(self) -> None:
        for student, teacher in zip(self.image_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher.mul_(self.teacher_momentum).add_(student, alpha=1 - self.teacher_momentum)
        for student, teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher.mul_(self.teacher_momentum).add_(student, alpha=1 - self.teacher_momentum)

    def forward(self, student_view: Tensor, teacher_view: Tensor) -> dict[str, Tensor]:
        student = self.student_head(self.image_encoder.get_global_embedding(student_view))
        with torch.no_grad():
            teacher = self.teacher_head(self.teacher_encoder.get_global_embedding(teacher_view))
        loss = -(functional.softmax(teacher / 0.04, dim=1) * functional.log_softmax(student / 0.1, dim=1)).sum(dim=1).mean()
        return {"loss": loss, "student": student, "teacher": teacher}
