"""Run the detection export/storage workflow end-to-end."""


def main() -> None:
    from .migrate_to_mongo import run_migration
    from .update_mongodb_geojson import update_geometries
    from .upload_masks import upload_all_masks

    print("[1/3] Migrating detections to MongoDB...")
    run_migration()

    print("\n[2/3] Updating MongoDB geometries from GeoJSON outputs...")
    update_geometries()

    print("\n[3/3] Uploading generated masks to Supabase...")
    upload_all_masks()


if __name__ == "__main__":
    main()
