# Official Code for ArgMatch

This repository contains the official implementation of the ICCV 2025 paper:

**ArgMatch: Adaptive Refinement Gathering for Efficient Dense Matching**
by *Yuxin Deng, Kaining Zhang, Linfeng Tang, Jiaqi Yang, and Jiayi Ma*.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ACuOoOoO/ArgMatch.git
cd ArgMatch
```

### 2. Create the Conda environment

Use the provided `env.yaml` file to create the Conda environment:

```bash
conda env create -f env.yaml
```

Then activate the environment:

```bash
conda activate argmatch
```

### 3. Install NATTEN

After activating the environment, install the NATTEN library:

```bash
pip3 install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels
```

**Warning:** ArgMatch relies on [NATTEN](https://github.com/SHI-Labs/NATTEN)’s *unfused neighborhood attention* API (the deprecated interface for computing query–key similarities). Recent versions of `natten` have removed this API, so you **must** use the compatible combination
`torch==2.3.0` and `natten==0.17.1+torch230cu121`.
We are working on a new version of ArgMatch that removes this dependency and will keep the codebase updated with future NATTEN releases.

---

## Pretrained Checkpoint

Download the pretrained ArgMatch checkpoint from:

* [Google Drive link](https://drive.google.com/file/d/1wHprz7nrZTrFxZmupaJH_TSndNdWsPkB/view?usp=sharing)

After downloading, place the checkpoint file `ckpt_clean.pth` in:

```text
ArgMatch/workspace/checkpoints/
```

The final path should look like:

```text
ArgMatch/
├── workspace/
│   ├── checkpoints/
│   │   ├── ckpt_clean.pth
```

---

## Data Preparation

Download **MegaDepth** and **ScanNet test** following the instructions from:

* [DKM](https://github.com/Parskatt/DKM)
* [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

This should result in the following data structure:

```text
ArgMatch/
├── data/
│   ├── megadepth/
│   │   ├── phoenix/
│   │   ├── Undistorted_SfM/
│   │   ├── prep_scene_info/
│   │   ├── 0015_0.1_0.3.npz ...
│   ├── scannet/
│   │   ├── scans_test/
│   │   ├── scannet_eval_list.txt
│   │   ├── test.npz
```

---

## Test

### Demo

We provide a simple demo script `experiments/demo.py` that:

* loads the pretrained ArgMatch checkpoint,
* takes two input images,
* computes dense correspondences,
* warps image B to image A, and
* saves visualizations (flow, certainty map, and concatenated images).

Run the demo with:

```bash
python -m experiments.demo \
  --checkpoint workspace/checkpoints/ckpt_clean.pth \
  --image_A assets/im_A.jpg \
  --image_B assets/im_B.jpg \
  --output_dir assets/outputs
```

After running, you should find the following files in `assets/outputs/`:

* `flow.png` – visualization of the dense flow field
* `certainty.png` – certainty / confidence map
* `concat.png` – concatenation of image A, image B, and the warped image B

### Evaluation

Before running the evaluation scripts, please create a directory for storing the results:

```bash
mkdir -p results
```

To reproduce the **relative camera pose estimation** results on **MegaDepth-1500**:

```bash
python -m experiments.test \
  --gpu 0 \
  --tests test_mega1500 \
  --sample_thresh 0.03
```

To reproduce the **relative camera pose estimation** results on **ScanNet**:

```bash
python -m experiments.test \
  --gpu 1 \
  --tests test_scannet \
  --sample_thresh 0.1
```

To reproduce the **dense matching** results on **MegaDepth-1500**:

```bash
python -m experiments.test \
  --gpu 2 \
  --tests test_megdepth_dense_scale
```

The corresponding logs and metrics will be saved under the `results/` directory.

---

## Training

We will release the training code in a future update.

---

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@inproceedings{deng2025argmatch,
  title     = {ArgMatch: Adaptive Refinement Gathering for Efficient Dense Matching},
  author    = {Yuxin Deng and Kaining Zhang and Linfeng Tang and Jiaqi Yang and Jiayi Ma},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## Acknowledgements

This implementation is heavily inspired by and builds upon the excellent work of [DKM](https://github.com/Parskatt/DKM) and [ROMA](https://github.com/Parskatt/roma). We sincerely thank the authors and maintainers of these projects for open-sourcing their code and ideas.

---

## License

This project is released under the terms of the license specified in the [LICENSE](LICENSE) file.
