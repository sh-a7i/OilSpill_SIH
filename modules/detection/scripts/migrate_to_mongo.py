import json
import os
import sys
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import GEOSPHERE, MongoClient  # type: ignore
from pymongo.errors import ConnectionFailure, PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ.get("MONGO_DB_NAME", "ocean_spill_intel")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "detections")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spill"
BASE_MASK_URL = os.environ["MASK_STORAGE_BASE_URL"]
METADATA_FILE = OUTPUT_DIR / "all_detections.json"


def run_migration() -> None:
    if not METADATA_FILE.exists():
        print(f"[ERROR] Could not find metadata file: {METADATA_FILE}")
        sys.exit(1)

    print("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        client.admin.command("ping")
        print("[SUCCESS] Connected to MongoDB Atlas cluster.")
    except ConnectionFailure as error:
        print(f"[ERROR] Could not connect to MongoDB: {error}")
        sys.exit(1)

    collection = client[DB_NAME][COLLECTION_NAME]
    with METADATA_FILE.open() as metadata_file:
        metadata_list = json.load(metadata_file)

    documents = []
    skipped_geometries = 0
    for item in metadata_list:
        image_id = item.get("image_id")
        if not image_id:
            continue

        geometry = None
        geojson_path = OUTPUT_DIR / f"{image_id}.geojson"
        if geojson_path.exists():
            try:
                geo_data = json.loads(geojson_path.read_text())
                if isinstance(geo_data, dict):
                    if "geometry" in geo_data:
                        geometry = geo_data["geometry"]
                    elif "coordinates" in geo_data and "type" in geo_data:
                        geometry = geo_data
            except (OSError, json.JSONDecodeError) as error:
                print(f"[WARN] Failed to parse GeoJSON for {image_id}: {error}")
                skipped_geometries += 1

        documents.append({
            "image_id": str(image_id),
            "has_spill": bool(item.get("has_spill", True)),
            "confidence": float(item.get("confidence", 1.0)),
            "area_pixels": int(item.get("area_pixels", 0)),
            "bbox": item.get("bbox", []),
            "geometry": geometry,
            "mask_url": f"{BASE_MASK_URL}/{image_id}_mask.png",
        })

    print(f"Parsed {len(documents)} documents (Skipped/Missing geometries: {skipped_geometries}).")
    try:
        collection.delete_many({})
        insert_result = collection.insert_many(documents)
        print(f"[SUCCESS] Uploaded {len(insert_result.inserted_ids)} detection records.")
        collection.create_index([("geometry", GEOSPHERE)])
        print("[SUCCESS] Spatial index created.")
    except PyMongoError as error:
        print(f"[ERROR] Database operation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()