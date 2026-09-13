# Member 1 — Satellite Detection Deliverable

## 1. Current Detection Architecture

The detection layer is implemented as a pre-trained U-Net segmentation pipeline using a ResNet-18 encoder. The model checkpoint is loaded directly from the local file `modules/detection/best_oil_spill_unet.pth`, and the main processing logic is defined in `modules/detection/pipeline.py`.

The current workflow is structured as follows:

- `modules/detection/evaluate.py` — evaluates the saved model on the validation split
- `modules/detection/infer.py` — runs inference on SAR images and generates aggregated metadata
- `modules/detection/scripts/run_all.py` — orchestrates the full end-to-end database/storage workflow
- `modules/detection/scripts/migrate_to_mongo.py` — stores detection metadata in MongoDB Atlas
- `modules/detection/scripts/update_mongodb_geojson.py` — updates geometry records in MongoDB
- `modules/detection/scripts/upload_masks.py` — uploads generated mask images to Supabase Storage

## 2. SAR Detection Principles

* **Why SAR?** Synthetic aperture radar provides all-weather, day/night detection capability and is well suited for ocean monitoring in cloud-covered regions.
* **Oil Slick Signature:** Oil dampens capillary waves, producing darker backscatter patches in SAR imagery.
* **Look-Alikes Handled:** Calm sea areas and some natural biogenic slicks are reduced using confidence filtering and minimum-area thresholding.

## 3. Model Baseline & Results

* **Architecture:** U-Net with ResNet-18 backbone
* **Dataset:** Deep-SAR Oil Spill Segmentation (Refined)
* **Input Resolution:** 256x256 grayscale SAR patches
* **Evaluation Metrics (Validation Set):**
  * **Val Dice:** 0.7604
  * **Val IoU:** 0.6509
  * **Val Loss:** 0.2403

The model is intended to be used as a pre-trained detector rather than retrained during normal deployment. The repository now emphasizes inference and pipeline execution using the saved checkpoint.

## 4. Current Data Storage Layout

The current system stores outputs in a hybrid local + cloud structure:

- **Local repository output:** `outputs/spill/all_detections.json`
  - This contains the aggregated detection metadata.
- **MongoDB Atlas:** detection metadata and geometry fields for spatial query use
- **Supabase Storage:** generated mask files (`*_mask.png`) and related binary artifacts

This means the repository no longer needs to keep every generated mask and GeoJSON file locally in the working tree.

## 5. How to Run the Detection Workflow

Run all commands from the repository root (`OilSpill_SIH`).

### Evaluate the pre-trained model

```bash
python -m modules.detection.evaluate evaluate --split val --limit 200
```

### Run inference

```bash
python -m modules.detection.infer infer --split val
```

This writes the aggregated metadata file to `outputs/spill/all_detections.json` by default.

### Run the complete pipeline workflow

```bash
python -m modules.detection.scripts.run_all
```

This runs the storage/database steps in order:

1. `python -m modules.detection.scripts.migrate_to_mongo`
2. `python -m modules.detection.scripts.update_mongodb_geojson`
3. `python -m modules.detection.scripts.upload_masks`

## 6. Database and Storage Setup

The project uses the following environment variables in the root `.env` file:

```bash
MONGO_URI="mongodb+srv://..."
MONGO_DB_NAME="ocean_spill_intel"
MONGO_COLLECTION_NAME="detections"
MASK_STORAGE_BASE_URL="https://<project>.supabase.co/storage/v1/object/public/spill-masks"
SUPABASE_URL="https://<project>.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
SUPABASE_BUCKET_NAME="spill-masks"
```

### MongoDB Atlas
- Stores detection metadata
- Stores geometry data for spatial querying

### Supabase Storage
- Stores generated binary mask files

## 7. Practical Notes for the Deliverable

* The pipeline is built around the saved pre-trained model and is designed for inference usage.
* The repository now emphasizes a clean split between local metadata output and cloud-backed heavy artifacts.
* The local output file `outputs/spill/all_detections.json` acts as the main repository-level summary of detections.
* Storage scripts are organized under `modules/detection/scripts` to keep the detection module self-contained.

## 8. Summary

This deliverable demonstrates an end-to-end SAR oil spill detection workflow using a pre-trained U-Net model, with metadata stored locally and spatial/binary outputs persisted in MongoDB Atlas and Supabase Storage. The design enables reproducible inference and scalable downstream use of the generated detection results.