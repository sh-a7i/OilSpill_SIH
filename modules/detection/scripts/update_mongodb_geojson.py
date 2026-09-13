import json
import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pymongo import GEOSPHERE, MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, OperationFailure, PyMongoError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "ocean_spill_intel"
COLLECTION_NAME = "detections"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spill"

# ----------------------------------------------------------------------
# GEOMETRY EXTRACTION HELPER
# ----------------------------------------------------------------------
def extract_clean_geometry(geo_data):
    if not geo_data or not isinstance(geo_data, dict):
        return None

    obj_type = geo_data.get("type")

    if obj_type in ["Polygon", "MultiPolygon", "Point", "LineString"]:
        return {"type": obj_type, "coordinates": geo_data.get("coordinates", [])}

    if obj_type == "Feature":
        return geo_data.get("geometry")

    if obj_type == "FeatureCollection":
        features = geo_data.get("features", [])
        if features:
            return features[0].get("geometry")

    if "geometry" in geo_data:
        return geo_data["geometry"]

    return None

# ----------------------------------------------------------------------
# BULK UPDATE ROUTINE
# ----------------------------------------------------------------------
def update_geometries():
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tlsCAFile=certifi.where(),
    )
    try:
        client.admin.command("ping")
    except PyMongoError as error:
        print(f"[ERROR] Could not connect to MongoDB: {error}")
        return

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch all records currently in MongoDB
    total_docs = collection.count_documents({})
    print(f"Found {total_docs} records in MongoDB collection '{COLLECTION_NAME}'.")

    bulk_operations = []
    operation_image_ids = []
    updated_count = 0
    missing_files = 0

    # Look up all .geojson files
    for doc in collection.find({}, {"image_id": 1}):
        img_id = doc["image_id"]
        
        # Check potential filename patterns (e.g., palsar_0.geojson)
        geojson_path = OUTPUT_DIR / f"{img_id}.geojson"
        
        if not geojson_path.exists():
            missing_files += 1
            continue

        try:
            with open(geojson_path, "r") as gf:
                raw_geo = json.load(gf)

            clean_geom = extract_clean_geometry(raw_geo)

            if clean_geom and clean_geom.get("coordinates"):
                bulk_operations.append(
                    UpdateOne(
                        {"image_id": img_id},
                        {"$set": {"geometry": clean_geom}}
                    )
                )
                operation_image_ids.append(img_id)
                updated_count += 1

        except Exception as e:
            print(f"[WARN] Failed to read {geojson_path.name}: {e}")

    # Execute updates in bulk for maximum speed
    if bulk_operations:
        print(f"Applying {len(bulk_operations)} geometry updates to MongoDB...")
        try:
            res = collection.bulk_write(bulk_operations, ordered=False)
            print(f"[SUCCESS] Matched: {res.matched_count}, Modified: {res.modified_count}")
        except BulkWriteError as error:
            details = error.details or {}
            write_errors = details.get("writeErrors", [])
            print(
                f"[WARN] Updated valid geometries but skipped "
                f"{len(write_errors)} invalid geometries."
            )
            for write_error in write_errors[:5]:
                image_id = operation_image_ids[write_error["index"]]
                print(
                    f"[WARN] Invalid geometry skipped for {image_id} "
                    f"(MongoDB error {write_error.get('code', 'unknown')})."
                )
    else:
        print("[ERROR] No matching .geojson files could be processed.")

    print(f"Summary: {updated_count} geometries attached, {missing_files} files not found locally.")

    # Re-apply 2dsphere index (sparse=True ignores any clean control scenes without polygons)
    print("Ensuring 2dsphere spatial index on 'geometry'...")
    try:
        collection.create_index([("geometry", GEOSPHERE)], sparse=True)
        print("[SUCCESS] Spatial index updated.")
    except OperationFailure as error:
        if error.code == 86:
            print("[WARN] Spatial index already exists; continuing.")
        else:
            raise

if __name__ == "__main__":
    update_geometries()