# ML process shown in this demo

## 1. Data representation

`synthetic_raw_imu.csv` contains simulated 50 Hz samples. Each row includes session ID, sample index/time, three acceleration values in m/s², three gyroscope values in degrees/s, a class label, and `synthetic=true` provenance.

## 2. Preprocessing

- Validate required fields and synthetic provenance.
- Convert sensor columns to numeric values and discard non-finite or incomplete rows.
- Keep each one-second session together as a group.
- Derive one feature row per session: acceleration norm deviation/variation/peak, gyro mean/RMS/peak, and peak jerk.
- Use median imputation in the model pipeline for any missing feature values.

## 3. Labels

- `Normal`: simulated light chassis vibration.
- `Turn`: simulated lateral acceleration and yaw rotation.
- `Bump`: simulated short moderate vertical shock.
- `Accident`: simulated stronger impulse and rotation.

These are generator-defined patterns, not a validated definition of a real crash.

## 4. Training and evaluation

- Random Forest classifier with fixed random seed.
- GroupShuffleSplit separates complete sessions into 75% train and 25% test sets.
- Confusion matrix, accuracy, balanced accuracy, precision, recall, F1, and feature importance are saved under `outputs/`.
- Because the samples are synthetic and generated from class-specific rules, high scores only show that the pipeline can learn those artificial patterns.

## 5. Hardware deployment status

Not done. The existing rover firmware still uses its hand-set prototype threshold rule. This package does not send predictions to the ESP32 dashboard. Also, the existing firmware does not read gyroscope axes, while this synthetic dataset includes simulated gyro values to match the illustrative table in the presentation.

## Flow

```text
Synthetic IMU samples
        ↓
Validation and cleaning
        ↓
One-second windows and features
        ↓
Session-separated train/test split
        ↓
Random Forest training
        ↓
Synthetic-only metrics and plots
```
