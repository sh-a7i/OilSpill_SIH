"""
Process detections from MongoDB (replaces the old all_detections.json +
local geojson file approach). Output format is UNCHANGED — same
spill_metadata.json structure M3/M5 already expect.

pip install pymongo python-dotenv shapely

Usage:
    python process_mongo_detections.py --top
    python process_mongo_detections.py            # process all
"""

import argparse
import json
import os
from shapely.geometry import shape, mapping
from dotenv import load_dotenv
from pymongo import MongoClient

from spill_characterisation import geodesic_area_km2, perimeter_km

load_dotenv()


def get_collection():
    client = MongoClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB_NAME"]]
    return db[os.environ["MONGO_COLLECTION_NAME"]]


def characterise_from_doc(doc):
    poly = shape(doc["geometry"])
    centroid = poly.centroid
    minx, miny, maxx, maxy = poly.bounds

    mrr = poly.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)
    edge1 = ((mrr_coords[0][0]-mrr_coords[1][0])**2 + (mrr_coords[0][1]-mrr_coords[1][1])**2) ** 0.5
    edge2 = ((mrr_coords[1][0]-mrr_coords[2][0])**2 + (mrr_coords[1][1]-mrr_coords[2][1])**2) ** 0.5
    length_deg, width_deg = max(edge1, edge2), min(edge1, edge2)

    import numpy as np
    m_per_deg_lat = 111.32
    m_per_deg_lon = 111.32 * np.cos(np.radians(centroid.y))
    length_km = length_deg * max(m_per_deg_lat, m_per_deg_lon)
    width_km = width_deg * max(m_per_deg_lat, m_per_deg_lon)

    return {
        "image_id": doc["image_id"],
        "detected": doc["has_spill"],
        "confidence": doc["confidence"],
        "centroid": [round(centroid.x, 5), round(centroid.y, 5)],
        "area_km2": round(geodesic_area_km2(poly), 2),
        "perimeter_km": round(perimeter_km(poly), 2),
        "length_km": round(length_km, 2),
        "width_km": round(width_km, 2),
        "bbox": [round(minx, 5), round(miny, 5), round(maxx, 5), round(maxy, 5)],
        "geometry": mapping(poly),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", action="store_true", help="Only the single highest-confidence detection")
    parser.add_argument("--id", type=str, default=None, help="Process one specific image_id instead of top/all")
    parser.add_argument("--out", type=str, default="outputs/geospatial/spill_metadata.json")
    args = parser.parse_args()

    collection = get_collection()
    query = {"has_spill": True}

    if args.id:
        doc = collection.find_one({**query, "image_id": args.id})
        if not doc:
            print(f"ERROR: image_id '{args.id}' not found")
            return
        results = [characterise_from_doc(doc)]
    elif args.top:
        doc = collection.find(query).sort("confidence", -1).limit(1)[0]
        results = [characterise_from_doc(doc)]
    else:
        docs = list(collection.find(query))
        results = [characterise_from_doc(d) for d in docs]

    out = results

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out} ({len(results)} spill(s) processed)")
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "geometry"}, indent=2))


if __name__ == "__main__":
    main()