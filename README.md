# Hypergraph Multi-Modal Learning for EEG-based Emotion Recognition in Conversation

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

  <p align="center">
    <a href="[https://arxiv.org/abs/2412.17337](https://arxiv.org/abs/2502.21154)">
      <img src="framework.png" alt="HyperMML" style="max-width: 80%; height: auto;"/>
    </a>
  </p>

## Usage
This repo is the official implementation of Hypergraph Multi-Modal Learning for EEG-based Emotion Recognition in Conversation.

## Abstract
Emotional Recognition in Conversation (ERC) is a valuable approach for diagnosing health conditions such as autism or depression, as well as understanding emotions in individuals who struggle to express their feelings. Current ERC methods primarily rely on complete semantic textual information, including audio and visual data, but face challenges in integrating physiological signals such as Electroencephalography (EEG). This research proposes a novel framework of Hypergraph Multi-Modal Learning (Hyper-MML), designed to effectively identify emotions in conversation by integrating EEG with audio and video information to capture complex emotional dynamics. Firstly, we introduce the module of Adaptive Brain Encoder with Mutual-cross Attention (ABEMA) of EEG signals, which captures emotion-relevant features across different frequency bands while adapting to subject-specific variations through hierarchical mutual-cross attention mechanisms. Secondly, we propose a module of Adaptive Hypergraph Fusion (AHFM), which actively models higher-order relationships between multi-modal signals in ERC. Experimental results validated on the EAV and AFFEC datasets demonstrate that our Hyper-MML significantly outperforms the state-of-art methods in ERC. The proposed Hyper-MML can serve as an effective communication tool for healthcare professionals, enabling better engagement with patients who have difficulty expressing their emotions.

## Data availability

Download EAV dataset: https://zenodo.org/records/13799131

Preprocessed .pkl files: Link: https://pan.baidu.com/s/19uQxD-v-0SNw9bwTgrjr0g Password: 6t6n 

Pre-trained model used for feature extraction: Link: https://pan.baidu.com/s/1DFd98GKIu6V2-7rnTarzIA Password: a1kz 

## Setting Up Your Environment and Dependencies
To ensure a clean and consistent working environment for this project, please follow these steps:

**1. Create and Activate a New Conda Environment**

First, create a dedicated Conda environment specifically for this project. This helps isolate project dependencies and prevents version conflicts with other projects.

```bash
conda create -n HyperMML python=3.8
```
Once the environment has been created, activate it using the following command:

```bash
conda activate HyperMML
```

**2. Install Required Dependencies**

With the environment activated, install all necessary dependencies using the `requirements.txt` file provided with the project. This file contains a list of all Python packages and their specific versions required for the project to function correctly.

```bash
pip install -r requirements.txt
```

**3. Verify Installation**

After the installation process completes, it's advisable to verify that all dependencies have been installed correctly. You can do this by checking the list of installed packages or by attempting to import key modules in a Python shell.

Additionally, you may want to list all Conda environments to ensure the environment was created and activated properly:

```bash
conda info --envs
```

By following these steps, you'll have a properly configured environment with all necessary dependencies installed, ensuring the project runs smoothly.



## Soft-computing extensions (added for SC lab)

All options are off by default, so the original Hyper-MML behaviour is unchanged. Code: `softcomputing.py`.

| Technique | Flag | What it does |
|---|---|---|
| Fuzzy-rough feature selection (rough sets + fuzzy sets) | `--rs_select quickreduct` | Finds a reduct per modality (EEG/audio/video) using the fuzzy-rough dependency degree; fitted on the training split only |
| Rough reduct via Genetic Algorithm (optimisation) | `--rs_select ga` | GA search maximising dependency degree while minimising reduct size, warm-started from QuickReduct |
| Rough boundary-region sample weighting | `--rs_weight` | Samples with low positive-region membership (ambiguous / noisy labels) get a smaller loss weight |
| Fuzzy TSK modality-reliability gate | `--fuzzy_gate` | 9-rule Takagi-Sugeno fuzzy system (certainty x agreement -> trust) re-weights each modality per sample before hypergraph fusion |
| Particle Swarm Optimisation | `tune_pso.py` / `modal_app.py::pso` | Tunes lr, l2, dropout, #hypergraph layers, rough-set sigma / reduct size, aux weight against **validation** F1 |

Use `--valid_frac 0.2` for model selection on a validation split. (The original script reports the best
test-set F1 over epochs; `RESULT {...}` now also prints test F1 at the best-validation epoch.)

Run on Modal: see the header of `modal_app.py` (`modal run modal_app.py::smoke`).
`make_synthetic.py` creates synthetic EAV-format features for pipeline checks only.

### Running on the open AFFEC dataset (no Baidu needed)

AFFEC (Zenodo 14794876, CC-BY) is openly downloadable. `preprocess/build_affec_features.py` builds Hyper-MML-format
features from the raw BIDS files (EEG band power, GSR stats, gaze/pupil stats; felt-valence -> 3 classes; trial-level
train/test split). These are hand-crafted features, **not** the paper's pretrained embeddings, so numbers are not
comparable with the paper.

    python -I preprocess/build_affec_features.py RAW_ROOT OUT_ROOT acl
    HYPERMML_ROOT=OUT_ROOT python train.py --Dataset AFFEC --subject 1 --valid_frac 0.2 --rs_select quickreduct

## Citation

If you find our work helpful to your research, we would appreciate a citation:

```bibtex
@article{kang2026hypergraph,
  title={Hypergraph multi-modal learning for EEG-based emotion recognition in conversation},
  author={Kang, Zijian and Li, Yueyang and Gong, Shengyu and Zeng, Weiming and Yan, Hongjie and Bian, Lingbin and Zhang, Zhiguo and Siok, Wai Ting and Wang, Nizhuan},
  journal={Neural Networks},
  pages={109057},
  year={2026},
  publisher={Elsevier}
}
```
