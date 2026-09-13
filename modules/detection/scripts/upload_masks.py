import os
from pathlib import Path

try:
    from supabase import create_client  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise SystemExit(
        "The Supabase client is not installed. Run: "
        "python -m pip install supabase"
    ) from exc
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
MASK_DIR = PROJECT_ROOT / "outputs" / "spill"
BUCKET_NAME = os.environ.get("SUPABASE_BUCKET_NAME", "spill-masks")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_all_masks() -> None:
    masks = list(MASK_DIR.glob("*_mask.png"))
    total = len(masks)
    print(f"Found {total} masks to upload...")

    for index, mask_path in enumerate(masks, 1):
        try:
            with mask_path.open("rb") as mask_file:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=mask_path.name,
                    file=mask_file,
                    file_options={"content-type": "image/png", "upsert": "true"},
                )
            if index % 100 == 0 or index == total:
                print(f"[{index}/{total}] Uploaded {mask_path.name}")
        except Exception as error:
            print(f"Error uploading {mask_path.name}: {error}")

    print("\nAll mask uploads finished!")


if __name__ == "__main__":
    upload_all_masks()