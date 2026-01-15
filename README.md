# Trajectory Alignment via Exploration-Based DPO in TextWorld

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**Authors:** Tejas Machkar, Mandar Kulkarni, Aashish Raheja

[cite_start]This repository contains the implementation of a **3-stage alignment pipeline** designed to solve long-horizon decision-making challenges in text-based games (TextWorld). [cite_start]By combining **Supervised Fine-Tuning (SFT)**, **Exploration-Based Trajectory Optimization (ETO)**, and **Direct Preference Optimization (DPO)**, we successfully align **Mistral-7B** agents to follow optimal trajectories without the need for scalar reward modeling or online reinforcement learning.

## 🚀 Key Features
* [cite_start]**3-Stage Pipeline:** A modular workflow moving from Imitation Learning (SFT) $\to$ Failure Discovery (ETO) $\to$ Preference Alignment (DPO).
* [cite_start]**Hard Negative Mining:** Automatic generation of "Hard Negative" preference pairs by contrasting SFT failures against Oracle trajectories.
* [cite_start]**Efficient Training:** Utilizes **QLoRA** (4-bit quantization) to fine-tune Mistral-7B on consumer hardware.
* [cite_start]**DPO Alignment:** Optimization of policy directly on preference data, eliminating the need for unstable reward models.

---

## 🏗️ Architecture

[cite_start]The pipeline consists of three interconnected stages:

1.  [cite_start]**Supervised Fine-Tuning (SFT):** The model mimics expert demonstrations from a Walkthrough Agent to establish a baseline policy.
2.  **Exploration-Based Optimization (ETO):** The SFT agent is deployed to explore. [cite_start]When it diverges from the Oracle, the divergence is captured as a "Rejected" trajectory, while the Oracle's path is "Chosen".
3.  [cite_start]**Direct Preference Optimization (DPO):** The model is trained on these `(chosen, rejected)` pairs to increase the likelihood of optimal actions and decrease the likelihood of known failure modes.

![System Architecture](https://via.placeholder.com/800x400?text=System+Architecture+Placeholder)
[cite_start]*(Refer to Figure 1 in the project report for the detailed pipeline diagram)* 

---

## 📊 Results

[cite_start]We evaluated the agents on **50 held-out TextWorld environments**. [cite_start]The DPO-aligned agent significantly outperforms the SFT baseline in both goal completion and navigational efficiency.

### Quantitative Performance

| Model | Avg Normalized Score (Higher is Better) | Avg Steps Taken (Lower is Better) |
| :--- | :---: | :---: |
| **SFT Baseline** | 0.7200 | 18.10 |
| **DPO Agent (Ours)** | **0.8200** | **14.28** |
[cite_start]*Source: *

### Behavioral Analysis

[cite_start]The DPO alignment successfully unlearned "looping" behaviors (e.g., repeatedly checking inventory), a common failure mode in LLM agents.

| Metric | SFT Baseline | DPO Agent | Improvement |
| :--- | :---: | :---: | :---: |
| **Repetition Rate** | 0.0590 | **0.0010** | **~98% Reduction** |
| **Hallucination Rate** | 0.1733 | 0.2531 | *See Note below* |
[cite_start]*Source: *

[cite_start]*> **Note:** While hallucination rate increased slightly, qualitative analysis shows this corresponds to the DPO agent attempting more complex, goal-oriented actions rather than passively looping safe commands.*

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/tmach22/TextWorld-DPO.git](https://github.com/tmach22/TextWorld-DPO.git)
cd TextWorld-DPO

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install textworld
