# Collect carla dataset (keypoints and 3D bboxes for cars)

## Prerequisites

* [Carla simulator](https://carla.readthedocs.io/en/stable/getting_started/)
* python3

## Usage

### 1 Test carla's PythonAPI

* Please change the path in file `test.py` to your own carla path.

```bash
python3 test.py
```

* If the vehicle type outputs correctly, the test passes

### 2 Generate traffic in Carla and capture dataset

```bash
python3 generate_traffic.py
```

* The captured results are automatically saved in `./data` by default.

* You can change the number and position of sensors, as well as the cars' number and types by adjusting parameters and the code. The optional parameters can be found by
```bash
python3 generate_traffic.py --help
```

### 3 Generate radar measurements form label

```bash
python3 generate_radar_meas_from_label.py <data/...>
```

* where `<data/...>` is the folder generated using `generate_traffic.py`, and the folder should contain `out_label`.

## Optional

* You can generate a custom trajectory using

```bash
python3 generate_custom_tracks.py
```

* where the spawn point and route are specified in the 70-72 lines of `generate_custom_tracks.py`. You can see the meaning of this points by visualizing the spaw point of the map.

```bash
python3 visulize_map.py
```
