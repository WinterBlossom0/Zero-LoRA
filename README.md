# Zero LoRA

Zero LoRA is a low-rank adapter that changes a frozen language model only by deleting its weights: during training
it may only shrink weights toward zero, and at the end each weight snaps to its original value or to exactly zero.
The goal is to find the parts of a model that cause a behaviour, at the cost of a LoRA.

This work is inspired by Deletion Tuning:

> Jack Lu, Zhenbang Yang, Mike Lasby, Tejas Pote, Yani Ioannou, Mengye Ren.
> *The Surprising Effectiveness of Deleting Weights in LLM Post-Training.* arXiv:2611.10866, 2026.
