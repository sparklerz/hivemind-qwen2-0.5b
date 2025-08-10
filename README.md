# Hivemind × Qwen2-0.5B-Instruct — Data-Parallel Fine-Tuning over the Internet

Fine-tuning **Qwen2-0.5B-Instruct** with **Hivemind** to coordinate multiple internet-connected GPUs (an initial peer on a public IP + “second” peers on free GPU notebooks) using a **TorchTune-based** training loop.

*At a glance (stack): Hivemind · PyTorch · TorchTune · Hugging Face Datasets/Transformers*

**TL;DR**: One always-on GPU hosts a public **DHT**; peers join from anywhere and train asynchronously. After a target global batch, parameters are averaged, and all peers continue from the globally averaged weights.

---

## Why this matters
High-speed interconnects (NVLink/IB) aren’t mandatory to collaborate on LLM fine-tuning. **Hivemind** uses a **Distributed Hash Table (DHT)**, allowing peers to join/leave and still converge via **asynchronous local updates + periodic parameter averaging**. This lowers the barrier for small teams/indies to run meaningful experiments.

---

## What’s in this repo
This is a **meta-repo**: a concise summary with links to the write-up and forks used for the experiment. Use the article for exact steps, configs, and logs.

---

## Links
- Write-up: *Finetuning Qwen 0.5B using Hivemind — Data Parallelism Over the Internet*  
  https://medium.com/@kannansarat9/finetuning-qwen-0-5b-using-hivemind-data-parallelism-over-the-internet-e20af1b15c05
- Base model: **Qwen/Qwen2-0.5B-Instruct**
  https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
- Fine-tuned Model: **hivemind-torchtune-Qwen2-0.5B**
  https://huggingface.co/ash001/hivemind-torchtune-Qwen2-0.5B
- Code forks used in this project:
    - TorchTune fork (configs/recipes): https://github.com/sparklerz/torchtune
    - Hivemind (modified for TorchTune integration): https://github.com/sparklerz/hivemind-modified-for-torchtune
