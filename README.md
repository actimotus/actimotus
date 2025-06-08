# Acti-Motus

Python-powered activity detection algorithms that build upon [Acti4](https://github.com/motus-nfa/Acti4), processing data from multiple accelerometers with a **requirement for a thigh-worn sensor**.

- Scientifically validated activity detection
- Device-independent, relies on RAW accelerometry
- Requires only a single accelerometer sensor worn on the thigh (front or side)
- Detects activities: lying, sitting, standing, walking, stair climbing, and bicycling
- An Optional back-worn sensor enhances lying and sitting detection
- A calf-worn sensor detects squatting and kneeling
- Posture analysis, measuring arm and trunk inclination
- Build upon [Acti4](https://github.com/motus-nfa/Acti4)
- Python | Multi-threaded

To learn more, read the user guide.

## Installation

Install using `pip install acti-motus`

## A Simple Example
```python
import pandas as pd
from acti_motus import Features, Activities

df = pd.read_parquet(thigh.parquet)
print(df.head(3))
#>                                      acc_x     acc_y     acc_z
#> datetime  
#> 2024-09-02 08:08:50.227000+00:00  0.218750 -0.171875 -0.773438
#> 2024-09-02 08:08:50.307000+00:00  0.257812 -0.203125 -0.937500
#> 2024-09-02 08:08:50.387000+00:00  0.242188 -0.226562 -0.953125

features = Features().extract(df)
acivities, references = Activities().detect(features)
#>                           activity  steps
#> datetime  
#> 2024-09-02 08:08:51+00:00      sit    0.0
#> 2024-09-02 08:08:52+00:00      sit    0.0
#> 2024-09-02 08:08:53+00:00      sit    0.0
```

## About Acti4

Developed by Jørgen Skotte, Acti4 was a sophisticated Matlab program designed to process data from multiple accelerometer sensors that participants wore on their thigh, hip, arm, and trunk. The core function of Acti4 was to classify physical activities, such as lying, sitting, standing, or walking. It also offered further calculations to assess a participant's posture by determining arm and trunk inclination. Lastly, these detections could be combined with participant diaries to obtain more contextual information, such as movement behaviour during periods of work and leisure.

The development of Acti4 concluded in July 2020 with its final release. Subsequently, the focus was redirected toward a successor project: rewriting the original Acti4 algorithm in Python. This new initiative, known as Motus, is being developed in partnership with SENS Innovation ApS.
