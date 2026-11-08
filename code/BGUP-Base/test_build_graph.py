"""BGUP-Base 全局构图的最小自包含检查。"""

import numpy as np
import networkx as nx

import Tools
from algorithm import total_weight


def test_build_graph_uses_all_online_agents():
    env = np.zeros((2, 4, 4), dtype=int)
    env[1, 0, 3] = 1
    env[0, 3, 0] = 1
    task_time = np.ones((4, 4), dtype=int)

    # [date, start, row, lin, end, type, speed, power, full_power]
    agents = np.array([
        [0, 0, 1, 1, 120, 1, 1, 0, 0],
        [0, 0, 2, 2, 120, 2, 1, 0, 10],
        [0, 0, 4, 4, 120, 3, 1, 100, 100],
    ], dtype=float)

    graph = Tools.build_graph(env, agents, task_time, 1, 10, 120)

    worker = "worker0|0,0"
    car = "car1|1,1"
    uav = "uav2|3,3"
    task = "task0,3"
    charging = "charging3,0"

    assert (worker, "None", uav, task, "None") in graph
    assert ("None", car, uav, "None", charging) in graph
    assert (worker, "None", "None", "None", "None") in graph
    assert ("None", car, "None", "None", "None") in graph
    assert ("None", "None", uav, "None", "None") in graph
    assert graph.has_edge(
        (worker, "None", uav, task, "None"),
        (worker, "None", "None", task, "None"),
    )
    assert graph.has_edge(
        ("None", car, uav, "None", charging),
        ("None", car, "None", "None", charging),
    )


def test_build_graph_does_not_scan_all_node_pairs():
    original = Tools.isConflict

    def fail_if_called(*_args):
        raise AssertionError("optimized build_graph must not call isConflict")

    Tools.isConflict = fail_if_called
    try:
        test_build_graph_uses_all_online_agents()
    finally:
        Tools.isConflict = original


def test_total_weight_is_order_independent():
    graph = nx.Graph()
    graph.add_nodes_from([
        ("a", {"weight": 1e16}),
        ("b", {"weight": 1.0}),
        ("c", {"weight": -1e16}),
    ])
    assert total_weight(graph, ["a", "b", "c"]) == 1.0
    assert total_weight(graph, ["c", "b", "a"]) == 1.0


def test_busy_and_faulty_agents_are_not_schedulable():
    env = np.zeros((2, 2, 2), dtype=int)
    env[1, 0, 0] = 1
    env[0, 1, 1] = 1
    task_time = np.ones((2, 2), dtype=int)
    agents = np.array([
        [0, 30, 1, 1, 120, 1, 1, 0, 0],
        [0, 0, 2, 1, 120, 3, 1, 100, 100],
    ], dtype=float)
    node = ("worker0|0,0", "None", "uav1|1,0", "task0,0", "None")

    assert node not in Tools.build_graph(env, agents, task_time, 1, 10, 120)
    agents[0, 1] = 0
    assert node not in Tools.build_graph(
        env, agents, task_time, 1, 10, 120, faulty_agents={0}
    )


def test_task_and_charge_durations_are_preserved():
    env = np.zeros((2, 2, 2), dtype=int)
    env[1, 0, 0] = 1
    env[0, 1, 1] = 1
    task_time = np.zeros((2, 2), dtype=int)
    task_time[0, 0] = 7
    task_agents = np.array([
        [0, 0, 1, 1, 120, 1, 1, 0, 0],
        [0, 0, 1, 1, 120, 3, 1, 100, 100],
    ], dtype=float)
    wut = [("worker0|0,0", "None", "uav1|0,0", "task0,0", "None")]

    task_result = Tools.updateEnv(
        env, task_agents, task_time, wut, 1, 10
    )
    assert task_result[1][0, 1] == 7
    assert task_result[1][1, 1] == 7
    assert task_result[6] == [7]

    charge_agents = np.array([
        [0, 0, 2, 2, 120, 2, 1, 0, 10],
        [0, 0, 2, 2, 120, 3, 1, 50, 100],
    ], dtype=float)
    cus = [("None", "car0|1,1", "uav1|1,1", "None", "charging1,1")]
    charge_result = Tools.updateEnv(
        env, charge_agents, np.zeros((2, 2)), cus, 1, 10
    )
    assert charge_result[1][0, 1] == 5
    assert charge_result[1][1, 1] == 5
    assert charge_result[5] == 1

    graph = Tools.build_graph(
        env, charge_agents, np.zeros((2, 2)), 1, 10, 4
    )
    assert cus[0] not in graph

    wt = [("worker0|0,0", "None", "None", "task0,0", "None")]
    assert Tools.updateEnv(
        env, task_agents, task_time.copy(), wt, 1, 10
    )[6] == []
    us = [("None", "None", "uav1|1,1", "None", "charging1,1")]
    assert Tools.updateEnv(
        env, charge_agents, np.zeros((2, 2)), us, 1, 10
    )[5] == 0


def test_charging_is_fcfs_and_includes_flight_energy():
    env = np.zeros((2, 1, 2), dtype=int)
    env[0, 0, 0] = 1
    agents = np.array([
        [0, 0, 1, 1, 120, 2, 1, 0, 10],
        [0, 0, 1, 2, 120, 3, 1, 50, 100],
        [0, 0, 1, 1, 120, 3, 1, 50, 100],
    ], dtype=float)
    queue = [
        ("None", "car0|0,0", "uav1|0,1", "None", "charging0,0"),
        ("None", "car0|0,0", "uav2|0,0", "None", "charging0,0"),
    ]
    result = Tools.updateEnv(
        env, agents, np.zeros((1, 2)), queue, 1, 10, 20
    )
    assert result[1][2, 1] == 5  # 先到的 UAV2 先充
    assert result[1][1, 1] == 11
    assert result[1][0, 1] == 11
    assert result[5] == 2

    limited = agents.copy()
    limited[1, 2:4] = [1, 1]
    limited_queue = [
        ("None", "car0|0,0", "uav2|0,0", "None", "charging0,0"),
        ("None", "car0|0,0", "uav1|0,0", "None", "charging0,0"),
    ]
    limited_result = Tools.updateEnv(
        env, limited, np.zeros((1, 2)), limited_queue, 1, 10, 5
    )
    assert limited_result[1][0, 1] == 5
    assert limited_result[5] == 1


def test_zero_distance_weights_are_finite():
    env = np.ones((2, 1, 1), dtype=int)
    agents = np.array([
        [0, 0, 1, 1, 120, 1, 1, 0, 0],
        [0, 0, 1, 1, 120, 2, 1, 0, 10],
        [0, 0, 1, 1, 120, 3, 1, 50, 100],
    ], dtype=float)
    graph = Tools.build_graph(env, agents, np.ones((1, 1)), 1, 10, 120)
    Tools.creatWeight(graph, env, agents, np.ones((1, 1)), 1, 10)
    assert all(np.isfinite(data["weight"]) for _, data in graph.nodes(data=True))


if __name__ == "__main__":
    test_build_graph_uses_all_online_agents()
    test_build_graph_does_not_scan_all_node_pairs()
    test_total_weight_is_order_independent()
    test_busy_and_faulty_agents_are_not_schedulable()
    test_task_and_charge_durations_are_preserved()
    test_charging_is_fcfs_and_includes_flight_energy()
    test_zero_distance_weights_are_finite()
    print("global graph check passed")
