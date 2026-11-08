import math


EPSILON = 1e-8
LOW_POWER_THRESHOLD = 0.30
SAFE_RESERVE_THRESHOLD = 0.20

DEFAULT_WEIGHTS = {
    "task": 0.25,
    "pair": 0.50,
    "charge": 0.15,
    "imbalance": 0.04,
    "move": 0.03,
    "energy_risk": 0.03,
}


def _distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _agent_position(agent_data, index):
    return int(agent_data[index, 2] - 1), int(agent_data[index, 3] - 1)


def _inside_region(agent_data, index, row, lin, scope):
    x, y = _agent_position(agent_data, index)
    return row <= x < row + scope and lin <= y < lin + scope


def _controlled_agents(agent_data, row, lin, scope, current_check_time, faulty_agents):
    controlled = {1: [], 2: [], 3: []}
    for index in range(agent_data.shape[0]):
        if index in faulty_agents:
            continue
        agent_type = int(agent_data[index, 5])
        if agent_type not in controlled:
            continue
        if not (agent_data[index, 1] <= current_check_time < agent_data[index, 4]):
            continue
        if _inside_region(agent_data, index, row, lin, scope):
            controlled[agent_type].append(index)
    return controlled


def _waiting_agents_at(agent_data, agent_type, position, current_check_time, faulty_agents):
    waiting = []
    for index in range(agent_data.shape[0]):
        if index in faulty_agents or int(agent_data[index, 5]) != agent_type:
            continue
        if current_check_time >= agent_data[index, 4]:
            continue
        if _agent_position(agent_data, index) == position:
            waiting.append(index)
    return waiting


def _nearest_distance(position, destinations):
    if not destinations:
        return float("inf")
    x, y = position
    return min(_distance(x, y, dx, dy) for dx, dy in destinations)


def _task_triples(
    env,
    agent_data,
    task_time,
    controlled,
    tasks,
    charging_points,
    current_time,
    current_check_time,
    time_limit,
    faulty_agents,
):
    triples = []
    controlled_workers = set(controlled[1])
    controlled_uavs = set(controlled[3])

    for task in tasks:
        waiting_workers = _waiting_agents_at(
            agent_data, 1, task, current_check_time, faulty_agents
        )
        waiting_uavs = _waiting_agents_at(
            agent_data, 3, task, current_check_time, faulty_agents
        )
        workers = controlled_workers | set(waiting_workers)
        uavs = controlled_uavs | set(waiting_uavs)
        nearest_charge = _nearest_distance(task, charging_points)
        if not math.isfinite(nearest_charge):
            continue
        duration = float(task_time[task[0], task[1]])

        for worker in workers:
            worker_controlled = worker in controlled_workers
            worker_position = _agent_position(agent_data, worker)
            if not worker_controlled and worker_position != task:
                continue

            worker_distance = _distance(*worker_position, *task)
            worker_speed = float(agent_data[worker, 6])
            if worker_speed <= 0:
                continue
            worker_start = max(current_time, float(agent_data[worker, 1]))
            worker_finish = worker_start + math.ceil(worker_distance / worker_speed + duration)

            for uav in uavs:
                uav_controlled = uav in controlled_uavs
                uav_position = _agent_position(agent_data, uav)
                if not uav_controlled and uav_position != task:
                    continue
                if not (worker_controlled or uav_controlled):
                    continue

                uav_speed = float(agent_data[uav, 6])
                full_power = float(agent_data[uav, 8])
                if uav_speed <= 0 or full_power <= 0:
                    continue

                uav_distance = _distance(*uav_position, *task)
                energy_required = math.ceil(uav_distance + nearest_charge)
                if energy_required > agent_data[uav, 7]:
                    continue

                uav_start = max(current_time, float(agent_data[uav, 1]))
                uav_finish = uav_start + math.ceil(uav_distance / uav_speed + duration)
                finish = max(worker_finish, uav_finish)
                deadline = min(
                    float(agent_data[worker, 4]),
                    float(agent_data[uav, 4]),
                    float(time_limit),
                )
                if finish > deadline:
                    continue

                triples.append(
                    {
                        "worker": worker,
                        "uav": uav,
                        "task": task,
                        "finish": finish,
                        "energy_ratio": energy_required / full_power,
                        "slack": deadline - finish,
                    }
                )

    return triples


def _greedy_disjoint_pair_count(triples):
    used_workers = set()
    used_uavs = set()
    used_tasks = set()
    selected = 0

    for triple in sorted(
        triples,
        key=lambda item: (item["finish"], item["energy_ratio"], item["slack"]),
    ):
        if triple["worker"] in used_workers:
            continue
        if triple["uav"] in used_uavs:
            continue
        if triple["task"] in used_tasks:
            continue
        used_workers.add(triple["worker"])
        used_uavs.add(triple["uav"])
        used_tasks.add(triple["task"])
        selected += 1

    return selected


def _charge_rescues(
    agent_data,
    controlled,
    charging_points,
    current_time,
    current_check_time,
    time_limit,
    faulty_agents,
):
    controlled_cars = set(controlled[2])
    controlled_uavs = set(controlled[3])
    waiting_cars = set()
    waiting_uavs = set()
    for charging in charging_points:
        waiting_cars.update(
            _waiting_agents_at(agent_data, 2, charging, current_check_time, faulty_agents)
        )
        waiting_uavs.update(
            _waiting_agents_at(agent_data, 3, charging, current_check_time, faulty_agents)
        )

    cars = controlled_cars | waiting_cars
    uavs = controlled_uavs | waiting_uavs
    rescued = set()
    urgency_by_uav = {}

    for uav in uavs:
        full_power = float(agent_data[uav, 8])
        if full_power <= 0:
            continue
        power = float(agent_data[uav, 7])
        power_ratio = power / full_power
        urgency = max(0.0, (LOW_POWER_THRESHOLD - power_ratio) / LOW_POWER_THRESHOLD)
        if urgency <= 0:
            continue
        urgency_by_uav[uav] = urgency

        uav_position = _agent_position(agent_data, uav)
        uav_speed = float(agent_data[uav, 6])
        if uav_speed <= 0:
            continue

        for charging in charging_points:
            uav_controlled = uav in controlled_uavs
            if not uav_controlled and uav_position != charging:
                continue
            uav_distance = _distance(*uav_position, *charging)
            if math.ceil(uav_distance) > power:
                continue

            for car in cars:
                car_controlled = car in controlled_cars
                car_position = _agent_position(agent_data, car)
                if not car_controlled and car_position != charging:
                    continue
                if not (uav_controlled or car_controlled):
                    continue

                car_speed = float(agent_data[car, 6])
                charge_rate = float(agent_data[car, 8])
                if car_speed <= 0 or charge_rate <= 0:
                    continue
                car_distance = _distance(*car_position, *charging)
                charge_time = math.ceil((full_power - power) / charge_rate)
                car_start = max(current_time, float(agent_data[car, 1]))
                uav_start = max(current_time, float(agent_data[uav, 1]))
                car_finish = car_start + math.ceil(car_distance / car_speed) + charge_time
                uav_finish = uav_start + math.ceil(uav_distance / uav_speed) + charge_time
                finish = max(car_finish, uav_finish)
                deadline = min(
                    float(agent_data[car, 4]),
                    float(agent_data[uav, 4]),
                    float(time_limit),
                )
                if finish <= deadline:
                    rescued.add(uav)
                    break
            if uav in rescued:
                break

    charge_score = sum(urgency_by_uav[uav] for uav in rescued)
    return rescued, charge_score


def _imbalance(triples):
    useful_workers = {triple["worker"] for triple in triples}
    useful_uavs = {triple["uav"] for triple in triples}
    return abs(len(useful_workers) - len(useful_uavs)) / (
        len(useful_workers) + len(useful_uavs) + EPSILON
    )


def _movement_cost(previous_center, row, lin, env, center_scope):
    if previous_center is None:
        return 0.0
    max_distance = math.sqrt(
        (env.shape[1] - center_scope) ** 2 + (env.shape[2] - center_scope) ** 2
    )
    if max_distance <= 0:
        return 0.0
    return _distance(previous_center[0], previous_center[1], row, lin) / max_distance


def _energy_risk(agent_data, controlled_uavs, charging_points, rescued_uavs):
    if not controlled_uavs:
        return 0.0
    total_risk = 0.0
    for uav in controlled_uavs:
        full_power = float(agent_data[uav, 8])
        if full_power <= 0:
            total_risk += 1.0
            continue
        power = float(agent_data[uav, 7])
        nearest_charge = _nearest_distance(_agent_position(agent_data, uav), charging_points)
        reserve_ratio = (power - nearest_charge) / full_power
        if reserve_ratio >= SAFE_RESERVE_THRESHOLD:
            risk = 0.0
        elif reserve_ratio < 0:
            risk = 1.0
        else:
            risk = (SAFE_RESERVE_THRESHOLD - reserve_ratio) / SAFE_RESERVE_THRESHOLD
        if uav in rescued_uavs:
            risk = 0.0
        total_risk += risk
    return total_risk / len(controlled_uavs)


def _region_features(
    env,
    agent_data,
    task_time,
    row,
    lin,
    center_scope,
    curr_time_age,
    single_step,
    time_limit,
    previous_center,
    faulty_agents,
):
    current_time = (curr_time_age - 1) * single_step
    current_check_time = curr_time_age * single_step
    controlled = _controlled_agents(
        agent_data, row, lin, center_scope, current_check_time, faulty_agents
    )
    tasks = list(zip(*((env[1] == 1).nonzero())))
    charging_points = list(zip(*((env[0] == 1).nonzero())))
    triples = _task_triples(
        env,
        agent_data,
        task_time,
        controlled,
        tasks,
        charging_points,
        current_time,
        current_check_time,
        time_limit,
        faulty_agents,
    )
    rescued_uavs, charge_score = _charge_rescues(
        agent_data,
        controlled,
        charging_points,
        current_time,
        current_check_time,
        time_limit,
        faulty_agents,
    )

    return {
        "row": row,
        "lin": lin,
        "task": len({triple["task"] for triple in triples}),
        "pair": _greedy_disjoint_pair_count(triples),
        "charge": charge_score,
        "imbalance": _imbalance(triples),
        "move": _movement_cost(previous_center, row, lin, env, center_scope),
        "energy_risk": _energy_risk(
            agent_data, controlled[3], charging_points, rescued_uavs
        ),
    }


def select_optimal_region_coarse(
    env,
    agent_data,
    task_time,
    center_scope,
    curr_time_age,
    single_step,
    time_limit,
    previous_center=None,
    faulty_agents=None,
    weights=None,
):
    """Choose the current MDC region by coarse score only; no look-ahead rollout."""
    if faulty_agents is None:
        faulty_agents = set()
    if weights is None:
        weights = DEFAULT_WEIGHTS

    row_count = env.shape[1] - center_scope + 1
    lin_count = env.shape[2] - center_scope + 1
    if row_count <= 0 or lin_count <= 0:
        raise ValueError("center_scope must not exceed the environment dimensions")

    candidates = []
    for row in range(row_count):
        for lin in range(lin_count):
            candidates.append(
                _region_features(
                    env,
                    agent_data,
                    task_time,
                    row,
                    lin,
                    center_scope,
                    curr_time_age,
                    single_step,
                    time_limit,
                    previous_center,
                    faulty_agents,
                )
            )

    positive_max = {
        name: max(candidate[name] for candidate in candidates)
        for name in ("task", "pair", "charge")
    }
    for candidate in candidates:
        task_score = candidate["task"] / (positive_max["task"] + EPSILON)
        pair_score = candidate["pair"] / (positive_max["pair"] + EPSILON)
        charge_score = candidate["charge"] / (positive_max["charge"] + EPSILON)
        candidate["score"] = (
            weights["task"] * task_score
            + weights["pair"] * pair_score
            + weights["charge"] * charge_score
            - weights["imbalance"] * candidate["imbalance"]
            - weights["move"] * candidate["move"]
            - weights["energy_risk"] * candidate["energy_risk"]
        )

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["score"],
            candidate["pair"],
            candidate["task"],
            candidate["charge"],
            -candidate["move"],
            -candidate["row"],
            -candidate["lin"],
        ),
    )
    return best["row"], best["lin"], best
