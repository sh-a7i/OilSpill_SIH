# Detection Pipelines

Use this module to run the existing detection pipeline with the pre-trained model at `modules/detection/best_oil_spill_unet.pth`.

Run all commands from the repository root (`OilSpill_SIH`).

## Quick start

### 1) Evaluate the pre-trained model

```bash
python -m modules.detection.evaluate evaluate --split val --limit 200
```

This loads the saved `.pth` checkpoint and prints mean Dice and IoU metrics.

### 2) Run inference on images

```bash
python -m modules.detection.infer infer --split val
```

This writes the combined metadata file to `outputs/spill` by default:

- `all_detections.json` aggregated detection metadata

In the current setup, the individual mask images and GeoJSON/geometry records are stored in Supabase Storage and MongoDB rather than kept as separate local files in the repo.

Optional overrides:

```bash
python -m modules.detection.infer infer --image-dir /path/to/images --output /path/to/output --weights /path/to/model.pth --threshold 0.5 --min-area 30 --geo-bounds 72.5,18.8,72.85,19.15
```

### 3) Run the full pipeline workflow

```bash
python -m modules.detection.scripts.run_all
```

This runs the three database/storage steps in order:

1. `python -m modules.detection.scripts.migrate_to_mongo`
2. `python -m modules.detection.scripts.update_mongodb_geojson`
3. `python -m modules.detection.scripts.upload_masks`

## Individual storage scripts

### MongoDB migration

```bash
python -m modules.detection.scripts.migrate_to_mongo
```

Reads `outputs/spill/all_detections.json`, connects to MongoDB Atlas, and stores detection metadata.

### Update MongoDB geometry records

```bash
python -m modules.detection.scripts.update_mongodb_geojson
```

Reads generated `.geojson` files and updates the MongoDB `geometry` field.

### Upload masks to Supabase Storage

```bash
python -m modules.detection.scripts.upload_masks
```

Uploads all `*_mask.png` files in `outputs/spill` to the configured Supabase bucket.

## Required environment variables

Your project root `.env` should include:

```bash
MONGO_URI="mongodb+srv://..."
MONGO_DB_NAME="ocean_spill_intel"
MONGO_COLLECTION_NAME="detections"
MASK_STORAGE_BASE_URL="https://<project>.supabase.co/storage/v1/object/public/spill-masks"
SUPABASE_URL="https://<project>.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
SUPABASE_BUCKET_NAME="spill-masks"
```

## Databases used

### MongoDB Atlas
- Stores detection metadata
- Stores geometry data for spatial querying

### Supabase Storage
- Stores generated mask images

## Output folder

By default, the local repository output is:

- `outputs/spill/all_detections.json`

This is the aggregated metadata file for the detections. The actual mask files are stored in Supabase Storage, and the geometry/metadata records are stored in MongoDB Atlas.