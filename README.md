# hello-world
This is a start repository.

Hello!Everyone!
I just start my science-road from now.
I hope that I could do some change for earth.

## Trajectory alignment script

This repository now includes a Python script for aligning two 2D trajectories
that contain `x`, `y`, and `yaw`:

- Script path: `scripts/align_trajectories.py`
- Input format: two CSV files with header `x,y,yaw`
- Correspondence rule: rows are matched by index

The script estimates a rigid transform in SE(2):

- position: `p_target ~= R(theta) * p_source + t`
- heading: `yaw_target ~= yaw_source + theta`

Unlike position-only alignment, this implementation uses both:

- x/y position correspondences
- yaw angle correspondences

to constrain the final transform.

### Example input

```csv
x,y,yaw
0.0,0.0,0.0
1.0,0.0,0.1
2.0,1.0,0.2
```

### Example usage

If yaw is in radians:

```bash
python3 scripts/align_trajectories.py source.csv target.csv --output aligned.csv
```

If yaw is in degrees:

```bash
python3 scripts/align_trajectories.py source.csv target.csv --degrees --output aligned.csv
```

You can also tune the relative importance of position and yaw constraints:

```bash
python3 scripts/align_trajectories.py source.csv target.csv \
  --position-weight 1.0 \
  --yaw-weight 2.0 \
  --output aligned.csv
```

### Output

The script prints:

- estimated rotation `theta`
- estimated translation `tx`, `ty`
- position RMSE after alignment
- yaw RMSE after alignment

If `--output` is provided, it also writes the aligned source trajectory to CSV.
