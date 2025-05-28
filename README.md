# Knowledge-Enriched ML for Tabular Data

This is the official repository for "Knowledge-Enriched Machine Learning for Tabular Data" (NeuS '25).
In this repository, you can find KE‑TALENT, a knowledge‑enriched tabular benchmark derived from TALENT, and reference implementations of four concept‑kernel–based approaches.


## How to Run

### Clone the repo and create the environment

```
$ git clone https://github.com/dalgu90/concept-kernels.git
$ cd concept-kernels
$ conda env create -f environment.yml
```

### Dataset (KE-TALENT)

Choose one of the following to obtain KE-TALENT:
- Option 1. Downloading KE-TALENT directly: Download the KE-TALENT archive [from this link (69MB)](https://drive.google.com/file/d/13PwmYb-GqfGzI7y7cpcR-g1XHCVqRYvo/view?usp=sharing), and unzip it under `data/talent/` so each dataset resides at, e.g. `data/talent/Abalone_reg/`.
- Option 2. Manual preprocessing: Download the original TALENT benchmark [from this repo](https://github.com/LAMDA-Tabular/TALENT/), place the unzipped folders under `data/talent/`, and execute notebooks `scripts/talent_data_preproc.ipynb` and `scripts/talent_column_embed.ipynb` to generate the 11 KE-TALENT datasets.

### Run concept kenrel approaches

Below is the commands to run the models with default hyper-parameters. Replace the argument "0" with the index of 11 datasets (0 to 10).
```
1. Smoothing (second arg: kernel_conv | rkhs_norm | laplacian)
$ bash run_talent_mlp.sh mlp_smooth kernel_conv mpnet 0

2. Value kernel
$ bash run_talent_mlp.sh mlp_gsp default mpnet 0 

3. Concept GAT
$ bash run_talent_gnn.sh gatconcept mpnet 0

4. Self-supervised learning (FT-Transformer)
$ bash run_talent_ssl4.sh ssl_ftt mpnet fixcorr InfoNCE false pretrain 0
$ bash run_talent_ssl4.sh ssl_ftt mpnet fixcorr InfoNCE false finetune 0
```

To perform hyper-parameter optimization (HPO), please run the below commands. Replace the argument "0" with the index of 11 datasets (0 to 10).
```
1. Smoothing (second arg: kernel_conv | rkhs_norm | laplacian)
$ bash hparam_search_talent_mlp.sh mlp_smooth kernel_conv mpnet 0 false

2. Value kernel
$ bash hparam_search_talent_mlp.sh mlp_gsp default mpnet 0 false

3. Concept GAT
$ bash hparam_search_talent_gnn.sh gatconcept2 mpnet 0 false gatconcept2

4. Self-supervised learning (FT-Transformer) - run once without HPO first
$ bash hparam_search_talent_ssl4.sh ssl_ftt mpnet fixcorr InfoNCE false 9 false
```

## Cite the work
```
@InProceedings{juyong2025knowledge,
  title = {Knowledge-Enriched Machine Learning for Tabular Data},
  author = {Kim, Juyong and Squires, Chandler and Ravikumar, Pradeep},
  booktitle = {International Conference on Neuro-symbolic Systems},
  year = {2025},
  volume = {288},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
