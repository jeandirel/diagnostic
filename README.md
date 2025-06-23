# Diagnostic Project

This project provides a small pipeline to compute the **Vertebral Heart Scale (VHS)** on veterinary chest radiographs using a U-Net segmentation model. It also includes utilities to generate diagnostic reports and a Streamlit dashboard for exploring the results.

## Prerequisites
- Python 3.8 or newer
- A dataset organised as described below
- Optional: a local Ollama instance with the `llama2` model for report generation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data layout
The code expects the following directory structure:

```
project/
├── data/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg / *.png
│   └── valid/
│       ├── _annotations.coco.json
│       └── *.jpg / *.png
└── generated_reports/
    └── pdf/                 # created automatically
```

## Usage
Below are the main scripts available in the repository.

### Training
Train the segmentation network on your dataset and save `unet_model.pth`:

```bash
python src/models/train.py
```

### Computing VHS
Run inference on the validation set and compute the VHS for each image. Results are stored in `vhs_results.csv`:

```bash
python scripts/compute_vhs.py
```

### Generating reports
Using the computed CSV, produce text reports and export them as PDFs:

```bash
python scripts/generate_reports_local.py      # produces vhs_llama2_reports.txt
python scripts/export_reports_to_pdf.py       # creates generated_reports/pdf/*.pdf
```

### End‑to‑end pipeline
A single script is provided to process all images and generate reports in one go:

```bash
python scripts/run_pipeline.py
```

### Dashboard
Visualise the results and upload new images with the Streamlit interface:

```bash
streamlit run scripts/dashboard.py
```

## Outputs
- `unet_model.pth` – weights of the trained U-Net
- `vhs_results.csv` – table containing VHS measurements
- `vhs_llama2_reports.txt` – textual reports generated with LLaMA 2
- `generated_reports/pdf/` – PDF reports for each processed image


