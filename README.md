<h1 align="center">
  <a href="https://github.com/acti-motus/acti-motus">
    <img src="https://github.com/acti-motus/acti-motus/blob/main/docs/acti-motus.png?raw=true" alt="Acti-Motus Logo" height="128px">
  </a>
</h1>

<div align="center">
  <a href="https://pypi.org/project/acti-motus/">
    <img src="https://img.shields.io/pypi/v/acti-motus" alt="PyPi Latest Release"/>
  </a>
  <a href="https://pypi.org/project/acti-motus/">
    <img src="https://img.shields.io/pypi/pyversions/acti-motus.svg" alt="Python Versions"/>
  </a>
  <a href="https://pepy.tech/projects/acti-motus">
    <img src="https://static.pepy.tech/badge/acti-motus/month" alt="Monthly Downloads"/>
  </a>
  <a href="#">
    <img src="#" alt="DOI Latest Release"/>
  </a>
  <a href="https://github.com/acti-motus/acti-motus/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/acti-motus/acti-motus.svg" alt="License"/>
  </a>
</div>

<div align="center">
  <p>Developed by the Danish <a href="https://nfa.dk/en">National Research Center for Working Environment (NRCWE)</a> in collaboration with <a href="https://www.sens.dk/en/">SENS Innovation ApS</a></p>
</div>


# Acti-Motus

Python-powered activity detection algorithms that build upon [Acti4](https://github.com/motus-nfa/Acti4), processing data from multiple accelerometers with a **requirement for a thigh-worn sensor**.

- Scientifically validated activity detection
- Device-independent, relies on RAW accelerometry
- Requires only a single accelerometer sensor worn on the thigh (front or side)
- Detects activities: lying, sitting, standing, walking, stair climbing, and bicycling
- An optional back-worn sensor enhances lying and sitting detection
- A calf-worn sensor detects squatting and kneeling
- Python

See [documentation](#) for more details.

## Installation

Install using `pip install acti-motus`.

## A Simple Example
```python
import pandas as pd
from acti_motus import Features, Activities, Exposures

df = pd.read_parquet(thigh.parquet)
print(df)
#>                                      acc_x        acc_y        acc_z
#> datetime  
#> 2024-09-02 08:08:50.227000+00:00  0.218750    -0.171875    -0.773438
#> 2024-09-02 08:08:50.307000+00:00  0.257812    -0.203125    -0.937500
#> 2024-09-02 08:08:50.387000+00:00  0.242188    -0.226562    -0.953125

features = Features().compute(df)
acivities, references = Activities().compute(features)
print(activities)
#>                           activity  steps
#> datetime  
#> 2024-09-02 08:08:51+00:00      sit    0.0
#> 2024-09-02 08:08:52+00:00      sit    0.0
#> 2024-09-02 08:08:53+00:00      sit    0.0

exposures = Exposures().compute(df)
print(exposures)
#>                                    sedentary           standing            on_feet
#> datetime  
#> 2024-09-02 00:00:00+00:00    0 days 09:12:21    0 days 04:34:03    0 days 01:26:00
#> 2024-09-03 00:00:00+00:00    0 days 17:05:21    0 days 04:11:19    0 days 01:30:02
#> 2024-09-04 00:00:00+00:00    0 days 18:26:01    0 days 04:05:18    0 days 00:46:19
#> 2024-09-05 00:00:00+00:00    0 days 04:47:29    0 days 00:59:53    0 days 00:08:28
```

Detailed information on Acti-Motus processing and features is available [here](#).

## About Acti4

Developed by Jørgen Skotte, Acti4 was a sophisticated Matlab program designed to process data from multiple accelerometer sensors that participants wore on their thigh, hip, arm, and trunk. The core function of Acti4 was to classify physical activities, such as lying, sitting, standing, or walking. It also offered further calculations to assess a participant's posture by determining arm and trunk inclination. Lastly, these detections could be combined with participant diaries to obtain more contextual information, such as movement behaviour during periods of work and leisure.

The development of Acti4 concluded in July 2020 with its final release. Subsequently, the focus was redirected toward a successor project: rewriting the original Acti4 algorithm in Python. This new initiative, known as Motus, is being developed in partnership with SENS Innovation ApS.

## Contributing

For guidance on setting up a development environment and how to make a contribution to Acti-Motus, see [Contributing to Acti-Motus](#).
