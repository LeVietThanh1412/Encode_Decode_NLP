
from pathlib import Path
from datasets import load_dataset

# Phiên bản chứa file Parquet đã được chuyển đổi
REVISION = "4f085df2ef57552b57ad74646b6c653ba2bbe6c9"

BASE_URL = (
    "https://huggingface.co/datasets/"
    f"IWSLT/mt_eng_vietnamese/resolve/{REVISION}/"
    "iwslt2015-en-vi"
)

# Tải trực tiếp Parquet, không chạy dataset script
data_files = {
    split: f"{BASE_URL}/mt_eng_vietnamese-{split}.parquet"
    for split in ["train", "validation", "test"]
}

print("Downloading IWSLT English-Vietnamese dataset...")

ds = load_dataset(
    "parquet",
    data_files=data_files
)

# Lưu cạnh file data.py
output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    return " ".join(text.split())

for split in ["train", "validation", "test"]:
    output_file = output_dir / f"{split}.tsv"
    count = 0

    with output_file.open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for row in ds[split]:
            en = clean_text(row["translation"]["en"])
            vi = clean_text(row["translation"]["vi"])

            if not en or not vi:
                continue

            f.write(f"{en}\t{vi}\n")
            count += 1

    print(f"{split}: {count:,} pairs -> {output_file}")

print("Dataset ready!")
