# OilTrace — Explainable Maritime Pollution Forensics

An end-to-end forensic pipeline that detects oil spills from satellite imagery, reconstructs their probable origin using ocean drift modelling, and ranks vessels based on evidence from AIS data.

Built for **Smart India Hackathon 2026**, OilTrace is a working prototype demonstrating the complete investigative pipeline on a controlled real-data scenario. It is not a production system or a legal attribution tool.

---

## The Problem

Illegal oil discharges at sea are difficult to investigate because:

* Oil spills drift with currents and wind, so the visible location may differ from the original discharge location.
* AIS data can contain gaps, coverage limitations, and other inconsistencies.
* Detection alone does not answer the critical investigative question: **which vessel may be responsible?**

OilTrace addresses this gap by connecting satellite-based detection with drift reconstruction and vessel intelligence to produce a ranked, evidence-backed list of investigation candidates.

---

## How It Works

```text
Satellite Imagery
       ↓
Oil Spill Detection
       ↓
Spill Characterisation
       ↓
Drift Simulation
   ↙           ↘
Backtracking   Forecast
   ↓
Probable Origin Zone
       ↓
AIS Correlation
       ↓
Evidence Scoring
       ↓
Ranked Vessel Candidates
       ↓
Investigation Dashboard
```

The system does not produce a definitive verdict. It generates an **explainable investigation lead**, with the evidence contributing to each vessel's ranking.

---

## Current Progress

### Detection

* U-Net with a ResNet-18 backbone
* Deep-SAR Oil Spill Segmentation (refined) dataset
* Validation: **Dice 0.76, IoU 0.65**
* Binary segmentation masks with GeoJSON and JSON metadata outputs
* Confidence-based filtering and area thresholding to reduce false positives

### Geospatial Characterisation

* Converts detected slicks into real-world geometry
* Calculates centroid, area, perimeter, length, width, and orientation
* Validated against raw detection masks
* Outputs standardised GeoJSON and JSON for downstream modules

### Drift, AIS, Attribution & Dashboard

Currently in development.

---

## Demo Scenario

**Region:** Arabian Sea corridor off Mumbai
**Coordinates:** 72.50°E, 18.80°N
**SAR scenes processed:** 1,615
**Detected slicks:** 1,548
**Clean-water controls:** 67

The controlled scenario is used to demonstrate the complete investigative workflow and module integration.

---

## Tech Stack

| Layer               | Technologies                        |
| ------------------- | ----------------------------------- |
| Detection           | PyTorch, U-Net, OpenCV              |
| Geospatial          | Shapely, GeoPandas, Rasterio, GeoPy |
| Coordinates         | WGS84 / EPSG:4326                   |
| Time                | UTC ISO-8601                        |
| Data Exchange       | GeoJSON, JSON                       |
| Vessel Intelligence | AIS data                            |

---

## Team Structure

| Member | Module                         |
| ------ | ------------------------------ |
| M1     | Oil Spill Detection            |
| M2     | Geospatial Characterisation    |
| M3     | Ocean Drift Modelling          |
| M4     | AIS / Vessel Intelligence      |
| M5     | Evidence Scoring & Attribution |
| M6     | Investigation Dashboard        |

---

## Why It Matters

Operational systems such as EMSA's CleanSeaNet demonstrate that satellite, oceanographic, and AIS data can support real-world maritime pollution investigations.

OilTrace focuses on building a **transparent, explainable, and modular prototype** that connects these stages into a single investigative workflow using publicly accessible data.

The current prototype establishes the core architecture. Future development will focus on improving detection accuracy, expanding data coverage, strengthening drift and attribution models, and increasing system robustness.

---

## Disclaimer

OilTrace is a research and demonstration prototype. Vessel rankings represent **investigation candidates based on available evidence** and should not be interpreted as proof of responsibility or used as an automated legal or enforcement decision.
