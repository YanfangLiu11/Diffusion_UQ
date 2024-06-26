[![DOI](https://zenodo.org/badge/788734734.svg)](https://zenodo.org/doi/10.5281/zenodo.12110815)

# Diffusion Uncertainty Quantification

This repository contains the implementation of a Conditional Generative Model developed to address specific challenges in uncertainty quantification using diffusion processes.

## Overview

This project aims to leverage deep learning methodologies, particularly generative models, to predict and quantify uncertainties in data-driven scenarios. The models are trained using a generative diffusion approach, suitable for complex datasets including bimodal distributions and experimental data from processes like ELM.

## Repository Structure

- `src/`: Contains the core Python scripts for the model and data processing.
- `data/`: Directory for storing example datasets used by the models.

## Getting Started

### Running the Code

To run the models, you can directly execute the Python scripts from the command line. Here is how you can do it:

#### For the ELM Data Model
Navigate to the directory containing the script and run:
```bash
python ELM.py
```

Before running the scripts, ensure you specify the correct paths to your data files in the scripts. For example, in ELM.py, set the data path as follows:

```
file_path = '/path/to/your/data/'
data = np.loadtxt(os.path.join(file_path, 'your_data_file.dat'))
```


