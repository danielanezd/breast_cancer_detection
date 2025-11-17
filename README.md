# Breast Cancer Detection

Documentation can be found [here](https://deepwiki.com/danielanezd/breast_cancer_detection)

# Prerequisites
1. An NVIDIA GPU
2. Java installed and added to your path
3. CBIS-DDSM dataset pulled in your local file system alongside the Cacl and Mass training and test csv files. Available in The Cancer Imaging Archive (here)[https://www.cancerimagingarchive.net/collection/cbis-ddsm/]
4. Install MLFlow with Docker and start the MLFlow Tracking Server

## Installation

1. Install [pyenv](https://github.com/pyenv/pyenv) 
2. Install python 3.13 with ```pyenv install 3.13``` and make it local with ```pyenv local 3.13```
3. Create virtualenv with ```python -m venv .venv```
4. Start virtualenv with ``` ./.venv/Scripts/activate```
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

## Runnin the conversion script

A script to convert the dicom images to 16-bit PNG was created in order to reduce the CPU bottleneck created by open dicom, parsing them and then feeding them to the models, this significantly reduced the training time from 4h to around 30-40min (with 50 epochs for ResNet50), so we suggest you do the same.

The command to run it will be:

```bash
python utils\conversion.py --source {RAW_IMAGES_PATH} --output {PATH_TO_STORE_NEW_IMAGES} --workers {NUM_OF_PARALLEL_WORKERS}
```

> **Note**: If you choose not to set workers, it will create as many workers as the 75% of your CPU cores.

> **Note 2**: To create train_png, val_png and test_png datasets, it's much easier to find and replace `.dcm` for `.png` in train, val, and test datasets using a text editor than to run a script. In case you need to create those again.

## Running the cleaning script

After converting to 16-bit PNGs you can remove text overlays and crop borders with:

```bash
python utils\cleaning.py --source {INPUT_DIR} --output {OUTPUT_DIR} [--gpu] [--crop 60]
```

- `--source/-s` and `--output/-o` are optional; they default to the paths defined in `utils/constants.py`.
- `--gpu` enables EasyOCR’s GPU mode (recommended for large batches).
- `--crop` controls how many pixels are trimmed from each border after masking text (default 60).

The script walks the source tree (handling PNG/JPEG/DICOM), loads each image via the 16-bit-safe reader (`utils/dicom.py`), masks detected text, optionally crops, and writes 16-bit PNGs to the output directory. A processing log (`log_<timestamp>_<source>.csv`) is saved alongside the cleaned images.

## Running the Streamlit Dashboard

To launch the interactive dashboard, run:

```bash
streamlit run dash/dashboard.py
```

This will start the Streamlit app. Open the provided local URL in your browser to interact with the dashboard for breast cancer detection.
