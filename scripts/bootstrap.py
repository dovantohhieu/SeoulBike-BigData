"""Tái tạo mô hình và kết quả cục bộ từ mã nguồn và dữ liệu UCI đã ghim."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Tải dữ liệu nếu cần, huấn luyện và tái tạo toàn bộ kết quả")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        return
    raw = ROOT / "data/raw/SeoulBikeData.csv"
    if not raw.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/download_data.py")], cwd=ROOT, check=True)
    expected = json.loads((ROOT / "data/DATA_SOURCE.json").read_text(encoding="utf-8"))["sha256"]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != expected:
        raise RuntimeError("CSV không khớp bản UCI đã ghim. Dừng trước khi huấn luyện.")
    subprocess.run([sys.executable, str(ROOT / "run_project.py"), "--all"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    print("Đã tạo mô hình, bảng kết quả, hình và demo HTML tại máy.")


if __name__ == "__main__":
    main()
