"""
m1_data/download_data.py  --  STEP 1: get the real datasets.

Run:  python -m m1_data.download_data          (METR-LA; add --all for PeMS-BAY too)

What we download (all public):
  * speed.csv      -- traffic speed matrix, one row per 5-minute step,
                      one column per sensor. Same numbers as the official
                      metr-la.h5 / pems-bay.h5 files, mirrored as CSV in
                      github.com/hazdzz/STGCN.
  * distances.csv  -- road-network driving distance between pairs of sensors
                      (from the DCRNN authors, github.com/liyaguang/DCRNN).
  * locations.csv  -- latitude / longitude of every sensor (for the map).
  * sensor_ids.txt -- the sensor id of each column of speed.csv.
"""
import urllib.request

import pandas as pd

from config import DATA_DIR

STGCN_URL = "https://raw.githubusercontent.com/hazdzz/STGCN/main/data"
DCRNN_URL = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"

FILES = {
    "metr-la": {
        "speed.csv": f"{STGCN_URL}/metr-la/vel.csv",
        "distances.csv": f"{DCRNN_URL}/distances_la_2012.csv",
        "locations.csv": f"{DCRNN_URL}/graph_sensor_locations.csv",
        "sensor_ids.txt": f"{DCRNN_URL}/graph_sensor_ids.txt",
    },
    "pems-bay": {
        "speed.csv": f"{STGCN_URL}/pems-bay/vel.csv",
        "distances.csv": f"{DCRNN_URL}/distances_bay_2017.csv",
        "locations.csv": f"{DCRNN_URL}/graph_sensor_locations_bay.csv",
    },
}


def download(name: str):
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES[name].items():
        target = folder / filename
        if target.exists():
            print(f"  [skip] {target} already exists")
            continue
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, target)

    # PeMS-BAY has no separate id file: its columns follow the (sorted)
    # sensor order of the locations file, so we create sensor_ids.txt from it.
    if name == "pems-bay" and not (folder / "sensor_ids.txt").exists():
        loc = pd.read_csv(folder / "locations.csv", header=None,
                          names=["sensor_id", "latitude", "longitude"])
        (folder / "sensor_ids.txt").write_text(",".join(loc.sensor_id.astype(str)))
        loc.to_csv(folder / "locations.csv", index=False)   # add a header row


if __name__ == "__main__":
    import sys
    from config import DATASET
    for dataset in (FILES if "--all" in sys.argv else [DATASET]):
        print(f"Dataset: {dataset}")
        download(dataset)
    print("Done. Files are in", DATA_DIR)
