import os
from tqdm import tqdm
import tempfile


def temp(x: str) -> str:
    return os.path.join(tempfile.gettempdir(), x)


def fetch_as_file(url):
    if url.startswith("/") or url.startswith("."):
        with open(url, "rb") as f:
            return f.read()
    import hashlib

    fp = temp(hashlib.md5(url.encode("utf-8")).hexdigest())
    download_file(url, fp, skip_if_exists=not os.getenv("NOCACHE"))
    return fp


def download_file(url, fp, skip_if_exists=True):
    import requests, pathlib

    if skip_if_exists and os.path.isfile(fp) and os.stat(fp).st_size > 0:
        return
    r = requests.get(url, stream=True)
    assert r.status_code == 200
    progress_bar = tqdm(
        total=int(r.headers.get("content-length", 0)),
        unit="B",
        unit_scale=True,
        desc=url,
    )
    (path := pathlib.Path(fp).parent).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        for chunk in r.iter_content(chunk_size=16384):
            progress_bar.update(f.write(chunk))
    f.close()
    os.rename(f.name, fp)
