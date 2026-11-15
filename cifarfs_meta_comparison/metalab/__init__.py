"""Meta-learning laboratory.

A small, self-contained PyTorch codebase for comparing four few-shot
meta-learning methods on CIFAR-FS:

    * MAML
    * Meta-SGD
    * Fixed-2   (pretrained backbone + 2-step classifier adaptation)
    * Meta-L2   (MAML with a meta-learned L2 anchor)

The package intentionally avoids higher-level meta-learning libraries so that
every gradient and every update rule is visible in ``metalab/algorithms.py``.
"""

__version__ = "1.0.0"
