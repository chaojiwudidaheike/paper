"""Generate the reproducible 50x50 large-scale MDC dataset."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


MAP_SIZE = 50
AGENT_COUNTS = {1: 560, 2: 120, 3: 320}
TASK_COUNT = 220
CHARGE_COUNT = 60
DEFAULT_SEED = 20260822
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Mat" / "Large.mat"


def generate(seed):
    rng = np.random.Generator(np.random.PCG64(seed))

    env = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    row_tiles = np.array_split(np.arange(MAP_SIZE), 10)
    col_tiles = np.array_split(np.arange(MAP_SIZE), 6)
    charge_cells = []
    for rows in row_tiles:
        for cols in col_tiles:
            charge_cells.append((rng.choice(rows), rng.choice(cols)))
    charge_cells = np.asarray(charge_cells, dtype=int)
    env[0, charge_cells[:, 0], charge_cells[:, 1]] = 1

    all_cells = np.arange(MAP_SIZE * MAP_SIZE)
    charge_indices = np.ravel_multi_index(charge_cells.T, (MAP_SIZE, MAP_SIZE))
    available_cells = np.setdiff1d(all_cells, charge_indices, assume_unique=True)
    task_indices = rng.choice(available_cells, size=TASK_COUNT, replace=False)
    task_rows, task_cols = np.unravel_index(task_indices, (MAP_SIZE, MAP_SIZE))
    env[1, task_rows, task_cols] = 1

    task_time = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    task_time[task_rows, task_cols] = 3

    agent_types = np.repeat(list(AGENT_COUNTS), list(AGENT_COUNTS.values()))
    agent_count = agent_types.size
    agents = np.zeros((agent_count, 9), dtype=np.float64)
    agents[:, 0] = 1
    agents[:, 1] = rng.integers(1, 121, size=agent_count)
    agents[:, 2:4] = rng.integers(1, MAP_SIZE + 1, size=(agent_count, 2))
    agents[:, 4] = agents[:, 1] + 60
    agents[:, 5] = agent_types

    speed_ranges = {1: (0.20, 0.30), 2: (0.50, 0.60), 3: (0.80, 0.90)}
    for agent_type, (low, high) in speed_ranges.items():
        mask = agent_types == agent_type
        agents[mask, 6] = np.round(rng.uniform(low, high, mask.sum()), 2)

    car_mask = agent_types == 2
    agents[car_mask, 8] = 10

    uav_mask = agent_types == 3
    full_power = rng.integers(30, 41, size=uav_mask.sum())
    agents[uav_mask, 8] = full_power
    agents[uav_mask, 7] = np.round(
        full_power * rng.uniform(0.70, 1.00, size=uav_mask.sum()), 2
    )

    return {
        "DayNum": np.array([[1]], dtype=np.uint8),
        "Env": env,
        "Order_Finall": agents,
        "Task_Time": task_time,
    }


def validate(data):
    day_num = data["DayNum"]
    env = data["Env"]
    agents = data["Order_Finall"]
    task_time = data["Task_Time"]

    assert day_num.shape == (1, 1) and day_num.dtype == np.uint8
    assert env.shape == (2, MAP_SIZE, MAP_SIZE) and env.dtype == np.uint8
    assert agents.shape == (sum(AGENT_COUNTS.values()), 9)
    assert agents.dtype == np.float64
    assert task_time.shape == (MAP_SIZE, MAP_SIZE) and task_time.dtype == np.uint8
    assert np.all((env == 0) | (env == 1))
    assert int(env[0].sum()) == CHARGE_COUNT
    assert int(env[1].sum()) == TASK_COUNT
    assert not np.any(env[0] & env[1])
    assert np.all(task_time[env[1] == 1] == 3)
    assert not np.any(task_time[env[1] == 0])
    assert day_num[0, 0] == 1 and np.all(agents[:, 0] == 1)
    assert np.all((agents[:, 1] >= 1) & (agents[:, 1] <= 120))
    assert np.all((agents[:, 2:4] >= 1) & (agents[:, 2:4] <= MAP_SIZE))
    assert np.all(agents[:, 4] - agents[:, 1] == 60)

    for agent_type, expected_count in AGENT_COUNTS.items():
        assert int(np.sum(agents[:, 5] == agent_type)) == expected_count

    worker = agents[:, 5] == 1
    car = agents[:, 5] == 2
    uav = agents[:, 5] == 3
    assert np.all((agents[worker, 6] >= 0.20) & (agents[worker, 6] <= 0.30))
    assert np.all(agents[worker, 7:9] == 0)
    assert np.all((agents[car, 6] >= 0.50) & (agents[car, 6] <= 0.60))
    assert np.all(agents[car, 7] == 0) and np.all(agents[car, 8] == 10)
    assert np.all((agents[uav, 6] >= 0.80) & (agents[uav, 6] <= 0.90))
    assert np.all((agents[uav, 8] >= 30) & (agents[uav, 8] <= 40))
    assert np.all(agents[uav, 7] <= agents[uav, 8])
    power_ratio = agents[uav, 7] / agents[uav, 8]
    assert np.all((power_ratio >= 0.70) & (power_ratio <= 1.00))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    data = generate(args.seed)
    validate(data)
    repeated = generate(args.seed)
    assert all(np.array_equal(data[key], repeated[key]) for key in data)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, data)
    loaded = {key: value for key, value in loadmat(output).items() if not key.startswith("__")}
    validate(loaded)
    assert all(np.array_equal(data[key], loaded[key]) for key in data)

    agents = data["Order_Finall"]
    print(f"Saved: {output}")
    print(f"Map: {MAP_SIZE}x{MAP_SIZE}; tasks={TASK_COUNT}; charges={CHARGE_COUNT}")
    print("Agents: workers=560; cars=120; uavs=320; total=1000")
    print(f"Online start: {agents[:, 1].min():.0f}-{agents[:, 1].max():.0f}; duration=60")
    print("Center_Scope is not stored; set Center_Scope=30 in both runners.")


if __name__ == "__main__":
    main()
