"""Tải hoặc xác thực nguồn UCI, kiểm tra SHA-256 trước khi ghi dữ liệu gốc."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
URL="https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip"

def run(verify_only=False):
    with urlopen(URL,timeout=60) as response:
        archive=response.read()
    with ZipFile(BytesIO(archive)) as z:
        matches=[name for name in z.namelist() if Path(name).name=="SeoulBikeData.csv"]
        if len(matches)!=1:raise RuntimeError("Không tìm thấy đúng một file dữ liệu UCI.")
        content=z.read(matches[0])
    expected=json.loads((ROOT/"data/DATA_SOURCE.json").read_text())["sha256"]
    digest=hashlib.sha256(content).hexdigest()
    if digest!=expected:
        raise RuntimeError("Dữ liệu tải về khác phiên bản thực nghiệm. Chưa ghi đè dữ liệu hiện tại.")
    target=ROOT/"data/raw/SeoulBikeData.csv"
    if verify_only:
        if not target.exists() or target.read_bytes()!=content:
            raise RuntimeError("CSV đang dùng khác dữ liệu chính thức vừa tải về.")
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    proof={"verified_at_utc":datetime.now(timezone.utc).isoformat(),"official_download":URL,
        "dataset_page":"https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        "bytes":len(content),"sha256":digest,"matches_pinned_sha256":True,
        "matches_packaged_raw_bytes":target.read_bytes()==content,
        "scope":"Exact byte equality to current UCI download; does not independently audit the original measurement process."}
    (ROOT/"data/SOURCE_VERIFICATION.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
    print("Đã xác thực CSV khớp hoàn toàn nguồn UCI và phiên bản SHA-256.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only",action="store_true")
    run(p.parse_args().verify_only)
