"""Download a large HTTP file in validated ranges and check its HF SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import requests

p = argparse.ArgumentParser()
p.add_argument("url")
p.add_argument("output")
p.add_argument("--workers", type=int, default=12)
a = p.parse_args()
head = requests.head(a.url, allow_redirects=False, timeout=30)
head.raise_for_status()
size = int(head.headers["X-Linked-Size"])
digest = head.headers["X-Linked-ETag"].strip('"')
out = Path(a.output)
parts = out.with_suffix(out.suffix + ".parts")
parts.mkdir(parents=True, exist_ok=True)
chunk = 16 * 1024 * 1024

def download(i):
    start, end = i * chunk, min(size, (i + 1) * chunk) - 1
    path = parts / str(i)
    if path.exists() and path.stat().st_size == end - start + 1:
        return i
    for attempt in range(5):
        try:
            with requests.get(a.url + f"?download=true&part={i}",
                              headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise RuntimeError(f"Unexpected range for part {i}: {r.status_code}, {r.headers.get('Content-Range')}")
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
                if temp.stat().st_size != end - start + 1:
                    raise RuntimeError(f"Short part {i}")
                temp.replace(path)
                return i
        except Exception as e:
            print(f"Part {i} retry {attempt + 1}: {type(e).__name__}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

count = (size + chunk - 1) // chunk
started = time.time()
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    futures = [pool.submit(download, i) for i in range(count)]
    for done, future in enumerate(as_completed(futures), 1):
        future.result()
        if done % 10 == 0 or done == count:
            print(f"{done}/{count} parts in {time.time() - started:.1f}s", flush=True)
hasher = hashlib.sha256()
temp_out = out.with_suffix(out.suffix + ".assembling")
with temp_out.open("wb") as target:
    for i in range(count):
        with (parts / str(i)).open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
                hasher.update(block)
assert hasher.hexdigest() == digest, "SHA-256 mismatch"
temp_out.replace(out)
for i in range(count):
    (parts / str(i)).unlink()
parts.rmdir()
print(f"Verified SHA-256 {digest}: {out}", flush=True)
