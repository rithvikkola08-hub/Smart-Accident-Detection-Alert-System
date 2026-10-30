"""Generate explicitly synthetic IMU samples for a classroom ML demo."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "synthetic_raw_imu.csv"
RNG = np.random.default_rng(20261028)
SAMPLE_RATE_HZ = 50
SAMPLES_PER_SESSION = SAMPLE_RATE_HZ  # one simulated second
SESSIONS_PER_CLASS = 120
CLASSES = ("Normal", "Turn", "Bump", "Accident")


def make_session(label: str, session_number: int) -> list[dict[str, object]]:
    """Produce one synthetic one-second session with class-specific patterns."""
    t = np.arange(SAMPLES_PER_SESSION, dtype=float) / SAMPLE_RATE_HZ
    phase = RNG.uniform(0, 2 * np.pi)
    # Resting gravity is along Z; small independent noise is present in every class.
    ax = RNG.normal(0, 0.35, SAMPLES_PER_SESSION)
    ay = RNG.normal(0, 0.35, SAMPLES_PER_SESSION)
    az = 9.81 + RNG.normal(0, 0.40, SAMPLES_PER_SESSION)
    gx = RNG.normal(0, 3.0, SAMPLES_PER_SESSION)
    gy = RNG.normal(0, 3.0, SAMPLES_PER_SESSION)
    gz = RNG.normal(0, 3.5, SAMPLES_PER_SESSION)

    if label == "Normal":
        # Mild chassis vibration and small acceleration changes.
        ax += 0.35 * np.sin(2 * np.pi * 4 * t + phase)
        az += 0.25 * np.sin(2 * np.pi * 6 * t + phase / 2)
    elif label == "Turn":
        direction = RNG.choice((-1.0, 1.0))
        ay += direction * (1.2 + 0.35 * np.sin(2 * np.pi * 1.2 * t + phase))
        gz += direction * (24 + 5 * np.sin(2 * np.pi * 1.2 * t + phase))
    elif label == "Bump":
        center = int(RNG.integers(12, 38))
        amplitude = RNG.uniform(3.0, 6.0)
        pulse = np.exp(-0.5 * ((np.arange(SAMPLES_PER_SESSION) - center) / 1.8) ** 2)
        az += amplitude * pulse
        ax += RNG.choice((-1.0, 1.0)) * amplitude * 0.18 * pulse
        gx += RNG.normal(0, 1, SAMPLES_PER_SESSION) + 15 * pulse
        gy += 10 * pulse
    else:  # Accident, simulated as a stronger short impulse and rotation.
        center = int(RNG.integers(12, 38))
        amplitude = RNG.uniform(9.0, 16.0)
        pulse = np.exp(-0.5 * ((np.arange(SAMPLES_PER_SESSION) - center) / 1.35) ** 2)
        axis = int(RNG.integers(0, 3))
        if axis == 0:
            ax += amplitude * pulse
        elif axis == 1:
            ay += amplitude * pulse
        else:
            az += amplitude * pulse
        gx += RNG.choice((-1.0, 1.0)) * 65 * pulse
        gy += RNG.choice((-1.0, 1.0)) * 75 * pulse
        gz += RNG.choice((-1.0, 1.0)) * 90 * pulse

    session_id = f"SYN-{label.upper()}-{session_number:03d}"
    rows = []
    for i in range(SAMPLES_PER_SESSION):
        rows.append({
            "session_id": session_id,
            "sample_index": i,
            "timestamp_ms": i * (1000 // SAMPLE_RATE_HZ),
            "ax_mps2": ax[i], "ay_mps2": ay[i], "az_mps2": az[i],
            "gx_dps": gx[i], "gy_dps": gy[i], "gz_dps": gz[i],
            "label": label,
            "synthetic": True,
        })
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in CLASSES:
        for n in range(SESSIONS_PER_CLASS):
            rows.extend(make_session(label, n))
    data = pd.DataFrame(rows)
    data.to_csv(OUT, index=False, float_format="%.5f")
    print(f"Wrote {len(data):,} synthetic rows to {OUT}")
    print("Synthetic samples per class:")
    print(data.groupby("label")["session_id"].nunique().to_string())


if __name__ == "__main__":
    main()
