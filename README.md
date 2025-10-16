# Breast Cancer Detection

Documentation can be found [here](https://deepwiki.com/danielanezd/breast_cancer_detection)

# Prerequisites
1. An NVIDIA GPU
2. Java installed and added to your path
3. CBIS-DDSM dataset pulled in your local file system alongside the Cacl and Mass training and test csv files. Available in The Cancer Imaging Archive (here)[https://www.cancerimagingarchive.net/collection/cbis-ddsm/]

## Installation

1. Install [pyenv](https://github.com/pyenv/pyenv) 
2. Install python 3.13 with ```pyenv install 3.13``` and make it local with ```pyenv local 3.13```
3. Create virtualenv with ```python -m venv .venv```
4. Install all dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running the Project

You can explore and experiment with the project using the provided Jupyter notebooks in the `notebooks/` directory. Open them with Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

## Running the Streamlit Dashboard

To launch the interactive dashboard, run:

```bash
streamlit run dash/dashboard.py
```

This will start the Streamlit app. Open the provided local URL in your browser to interact with the dashboard for breast cancer detection.
