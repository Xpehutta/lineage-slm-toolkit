"""QLoRA configuration with the reasons behind each value.

Defaults match the run whose log is in the article: Qwen3-8B in 4-bit,
rank 16, lr 2e-4, three epochs, 8192 tokens, batch 1 with 8 accumulation steps
(43.6M trainable parameters, 0.53% of the base). Packing is off because JSON
answers must not be spliced across an example boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class QLoRAConfig:
    base_model: str = "unsloth/Qwen3-8B-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 8192
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    warmup_steps: int = 15
    lr_scheduler: str = "linear"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    packing: bool = False
    seed: int = 1

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        return asdict(self)
