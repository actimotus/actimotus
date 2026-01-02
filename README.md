[![PyPi](https://img.shields.io/pypi/v/acti-motus.svg)](https://pypi.org/project/acti-motus/)
[![Coverage](https://img.shields.io/pypi/pyversions/acti-motus.svg)](https://pypi.org/project/acti-motus/)
[![Monthly Downloads](https://static.pepy.tech/badge/acti-motus/month)](https://pepy.tech/projects/acti-motus)
[![License](https://img.shields.io/github/license/acti-motus/actimotus.svg)](https://github.com/actimotus/actimotus/blob/main/LICENSE)

A Python-powered human activity recognition algorithm building upon [Acti4](https://github.com/motus-nfa/Acti4). It processes data from multiple accelerometers with a primary **requirement for a thigh-worn sensor**.

- Scientifically validated activity detection.
- Device-independent, relies on RAW accelerometry.
- **Minimum Requirement:** Single accelerometer worn on the thigh (front or side).
- **Detects:** Lying, sitting, standing, walking, stair climbing, and bicycling.
- **Optional:** A back-worn sensor enhances lying/sitting detection; a calf-worn sensor adds squatting/kneeling detection.

## About predecessor Acti4
Developed by the Danish [National Research Center for Working Environment (NRCWE)](https://nfa.dk/en), **Acti4** was a MATLAB-based tool designed to classify physical activities (lying, sitting, standing, walking) and assess posture using sensors on the thigh, hip, arm, and trunk. It allowed for combining detections with participant diaries to analyze movement behavior during work and leisure. Development of Acti4 concluded in July 2020. Focus has since shifted to its Python-based successor, **ActiMotus**, which is being developed in partnership with [SENS Innovation ApS](https://www.sens.dk/en/) and is the core of **Motus** infrastructure.

## Installation
Install using `pip install acti-motus`.

## Usage example
To see ActiMotus in action, here is a simple workflow processing data from a single thigh-worn accelerometer.

### 1. Load the raw accelerometer data

```python
import pandas as pd
from actimotus import Sens

# Load binary data
raw = Sens.from_bin(raw_thigh.bin)
print(raw)

datetime                           acc_x       acc_y       acc_z  
2024-09-02 08:08:50.227000+00:00   0.218750   -0.171875   -0.773438
2024-09-02 08:08:50.307000+00:00   0.257812   -0.203125   -0.937500
2024-09-02 08:08:50.387000+00:00   0.242188   -0.226562   -0.953125
2024-09-02 08:08:50.467000+00:00   0.234375   -0.242188   -0.945312
2024-09-02 08:08:50.548000+00:00   0.257812   -0.226562   -0.953125
```

### 2. Extract features and detect activity types
Next step is to extract features and detect activity types:

```python
from actimotus import Features, Activities

# Calculate features from the raw data
features = Features(calibrate=False).compute(df)

# Classify activities
acivities, references = Activities(vendor="Sens").compute(features)
print(activities)

datetime                    activity  
2024-09-02 08:08:51+00:00   sit
2024-09-02 08:08:52+00:00   sit
2024-09-02 08:08:53+00:00   sit
2024-09-02 08:08:54+00:00   sit
2024-09-02 08:08:55+00:00   sit
```

### 3. Generate aggregated exposures
To summarize the detected activities over time (e.g., daily totals), transform the data into exposures:

```python
from actimotus import Exposures

# Compute daily exposures from the activity data
exposures = Exposures().compute(df)
print(exposures)

datetime                    sit               stand             walk              bicycle
2024-09-02 00:00:00+00:00   0 days 04:13:20   0 days 02:29:48   0 days 01:43:48   0 days 00:00:00
2024-09-03 00:00:00+00:00   0 days 07:20:48   0 days 02:23:44   0 days 01:42:22   0 days 00:17:35
2024-09-04 00:00:00+00:00   0 days 08:16:13   0 days 02:24:17   0 days 00:54:27   0 days 00:37:01
2024-09-05 00:00:00+00:00   0 days 00:44:01   0 days 00:38:16   0 days 00:10:19   0 days 00:17:27
```

Detailed information on ActiMotus processing and features is available in the [learning center](https://actimotus.josefheidler.cz/learn).
