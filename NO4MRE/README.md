# NO4MRE

**NO4MRE** is a benchmarking framework for evaluating neural operators on Magnetic Resonance Elastography (MRE) inversion tasks. It includes high-fidelity synthetic MRE simulations, data preparation tools, and machine learning baseline scripts. This repository is designed to support research in physics-informed and data-driven operator learning.

---

## 📁 Data Preparation

1. Download the dataset from our [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EQ6UKN).
2. Unzip the `processed_data.zip` file and place its contents under the `simulation/` directory.
3. Download and move the following files to the same `simulation/` folder:
   - `data_general_heter.pkl`
   - `data_general_homo.pkl`

Your directory structure should look like this:

```
NO4MRE/
├── simulation/
│ ├── processed_data/
│ ├── data_general_heter.pkl
│ └── data_general_homo.pkl
```


## 🧪 Simulation with FEniCSx (DolfinX)

We provide a Docker image for running high-fidelity elasticity simulations using DolfinX with complex-number support.

### Step 1: Pull and Run the Docker Image

```
cd NO4MRE
docker pull weihengz/dolfinx_mre:latest
docker run -it -v "$PWD:/workspace" weihengz/dolfinx_mre:latest
```

### Step 2:  Activate Complex Mode for Dolfinx and run the data generator

```
source /usr/local/bin/dolfinx-complex-mode
cd /workspace/simulation/
python fem_img_complex.py
python fem_img_complex_incom.py
```


## 🧪 Machine learning baselines

### step 1: create a conda environment with Python=3.12 and install the required package of requirement.txt:
```
conda create -n no4mre python=3.12
conda activate no4mre
pip3 install -r requirements.txt
```

### step 2: run the bash script to run data-driven training of the models and physics-informed training of the models:
```
bash scripts/NO_heter.sh
bash scripts/PINO_heter.sh
```

✨ If you use this dataset or codebase in your research, please consider citing us:
```
@dataset{zhong_no4mre_2025,
  author       = {Weiheng Zhong, Matthew Urban, Hadi Meidani},
  title        = {{NO4MRE: Neural Operator Benchmarking for Magnetic Resonance Elastography}},
  year         = {2025},
  publisher    = {Harvard Dataverse},
  doi          = {10.7910/DVN/EQ6UKN},
  url          = {https://doi.org/10.7910/DVN/EQ6UKN}
}
```



