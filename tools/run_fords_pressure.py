"""Call the unmodified upstream function in an isolated import namespace."""
import sys
from pathlib import Path
import numpy as np


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/fords/offline"))
    from topo_factory import downscale_air_pressure
    folder = Path(sys.argv[1])
    with np.load(folder / "input.npz", allow_pickle=False) as a:
        out = downscale_air_pressure(a["pressure"], a["temperature_reference"], a["temperature"],
                                    a["elevation_reference"], a["elevation"])
    np.save(folder / "output.npy", out, allow_pickle=False)


if __name__ == "__main__":
    main()
