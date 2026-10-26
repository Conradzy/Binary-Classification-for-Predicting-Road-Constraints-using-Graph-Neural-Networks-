# Binary Classification for Predicting Road Constraints Using Graph Neural Networks

## Overview

This repository contains a Google Colab notebook for identifying constrained road segments in Panabo City, Philippines. It represents each road segment as a graph node and connects segments that share endpoints, allowing graph neural networks to learn from both road attributes and network structure.

The notebook performs two related tasks:

- binary classification: predict whether a road segment is normal or constrained;
- score estimation: predict a continuous constraint score for downstream mapping and routing.

Examples of modeled constraints include narrow roads, traffic-prone areas, school zones, night-market traffic, and road construction.

## Notebook Workflow

1. Load the Panabo road network, barangay boundaries, and manually prepared constraint layers.
2. Match constraint annotations to selected road geometries.
3. Assign barangays and engineer spatial, geometric, topological, and proximity features.
4. Create spatial train, validation, and test blocks to reduce geographic leakage.
5. Build a road-segment graph in which adjacent segments share endpoints.
6. Train and compare GCN, GAT, and GraphSAGE models.
7. Evaluate classification, score-estimation, and inference-time metrics.
8. Analyze GAT decision thresholds and export predicted constraints and routing weights.

## Models

The notebook compares three PyTorch Geometric architectures:

- **GCN** — Graph Convolutional Network;
- **GAT** — Graph Attention Network with four attention heads in its first layer;
- **GraphSAGE** — neighborhood-aggregation model.

Each architecture has a classification head and a regression head. Training uses a fixed random seed (`42`), class weighting, early stopping, and validation-based threshold selection. The notebook also evaluates a manually selected GAT threshold for recall-sensitive routing use cases.

## Data Requirements

The geospatial datasets are not included in this repository. Before running the notebook, place the required files in the Google Drive data directory configured near the top of the notebook.

Core inputs include:

- `roads_panabo.gpkg` — base road-segment network;
- `PanaboBarangayBoundaries.geojson` — barangay boundaries;
- `narrow_roads_part*.gpkg` — selected narrow-road annotations;
- `new_panabo_nightmarket_4pm_to_12am.gpkg` — night-market constraint layer.

The workflow can also use the prepared construction, market, school, highway, and general traffic GeoPackage layers referenced in the notebook. Keep their filenames consistent with the code.

## How to Run

1. Open [`final_najud_gnn_gattt.ipynb`](final_najud_gnn_gattt.ipynb) in Google Colab.
2. Enable a GPU runtime if one is available: **Runtime > Change runtime type > T4 GPU**.
3. Put the input datasets in Google Drive. The notebook currently expects:

   ```text
   /content/drive/MyDrive/bfp_capstone_newrc/
   ```

4. Update `DATA_DIR` and `OUTPUT_DIR` in the path-setup cell if your folder differs.
5. Run the cells from top to bottom. The first cell installs the Python packages used by the workflow.

Main dependencies include Python 3, PyTorch, PyTorch Geometric, GeoPandas, NetworkX, scikit-learn, SciPy, Shapely, pandas, NumPy, Matplotlib, Seaborn, Folium, and Mapclassify.

## Outputs

The notebook writes generated artifacts to the configured `OUTPUT_DIR`, including:

- trained GCN, GAT, and GraphSAGE checkpoints;
- model-comparison and threshold-sensitivity CSV files;
- evaluation charts and a confusion matrix;
- predicted-constraint GeoPackage and GeoJSON layers;
- a GAT checkpoint with its selected manual threshold;
- `edge_weights.json` and related files for ALT-based routing.

Generated data, model files, and figures are intentionally not stored in this repository.

## Project Structure

```text
.
|-- README.md
`-- final_najud_gnn_gattt.ipynb
```

## Notes

- The notebook is designed for Google Colab and mounts Google Drive for input and output storage.
- Road data is handled in EPSG:4326 and projected to EPSG:32651 for distance and geometry calculations.
- Exact selected-geometry matching is used for narrow-road annotations to avoid labeling nearby, unselected segments.
- Predictions should be validated with local field knowledge before they are used in operational or emergency routing.
