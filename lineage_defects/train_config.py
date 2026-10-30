"""QLoRA configuration with the reasons behind each value.

Defaults match the article: Qwen3-8B in 4-bit, rank 16, lr 2e-4, three epochs,
4096 tokens, packing off. Packing is off because JSON answers must not be
spliced across an example boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class QLoRAConfig:
    base_model: str = "unsloth/Qwen3-8B-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    packing: bool = False
    seed: int = 42

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        return asdict(self)
