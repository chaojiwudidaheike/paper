"""Focused checks for the two region-aware BGUP-Cache algorithms."""

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent


def load_tools(directory, name):
    path = CODE_ROOT / directory / "Tools.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    env = np.zeros((2, 4, 4), dtype=int)
    env[1, 0, 0] = env[1, 3, 3] = 1
    env[0, 0, 1] = 1
    task_time = np.ones((4, 4), dtype=int)
    agents = np.array([
        [0, 0, 1, 1, 120, 1, 1, 0, 0],
        [0, 0, 2, 1, 120, 2, 1, 0, 10],
        [0, 0, 2, 2, 120, 3, 1, 100, 100],
        [0, 0, 4, 4, 120, 1, 1, 0, 0],
    ], dtype=float)
    return env, agents, task_time


def check(directory, name):
    tools = load_tools(directory, name)
    env, agents, task_time = fixture()
    graph = tools.build_graph(
        env, agents, task_time, 0, 0, 2, 1, 10, 120
    )

    assert any("worker0|0,0" in node for node in graph)
    original = tools.calculate_distance
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    tools.calculate_distance = counted
    try:
        tools.creatWeight(graph, env, agents, task_time, 1, 10)
    finally:
        tools.calculate_distance = original
    assert len(calls) == len(set(calls))


def test_new_bgup_cache():
    check("cpol-ashs-new-BGUP-Cache", "new_bgup_cache_tools")


def assert_preserves_online_boundary_and_async_nodes(
        original_directory, cached_directory, name_prefix):
    original = load_tools(
        original_directory, f"{name_prefix}_original_tools"
    )
    cached = load_tools(
        cached_directory, f"{name_prefix}_cached_tools"
    )

    env = np.ones((2, 1, 1), dtype=int)
    task_time = np.ones((1, 1), dtype=int)
    agents = np.array([
        [0, 10, 1, 1, 120, 1, 1, 0, 0],
        [0, 10, 1, 1, 120, 3, 1, 100, 100],
    ], dtype=float)
    base = original.build_graph(
        env, agents, task_time, 0, 0, 1, 1, 10, 120
    )
    optimized = cached.build_graph(
        env, agents, task_time, 0, 0, 1, 1, 10, 120
    )
    assert set(base) == set(optimized)
    assert set(base.edges) == set(optimized.edges)

    env = np.zeros((2, 2, 2), dtype=int)
    env[1, 1, 1] = 1
    env[0, 0, 0] = 1
    task_time = np.ones((2, 2), dtype=int)
    agents = np.array([
        [0, 0, 2, 2, 120, 1, 1, 0, 0],
        [0, 0, 1, 1, 120, 3, 1, 100, 100],
    ], dtype=float)
    base = original.build_graph(
        env, agents, task_time, 0, 0, 1, 1, 10, 120
    )
    optimized = cached.build_graph(
        env, agents, task_time, 0, 0, 1, 1, 10, 120
    )
    assert set(base) == set(optimized)
    assert set(base.edges) == set(optimized.edges)


def test_new_preserves_online_boundary_and_async_nodes():
    assert_preserves_online_boundary_and_async_nodes(
        "cpol-ashs-new", "cpol-ashs-new-BGUP-Cache", "new"
    )


def test_coarse_bgup_cache():
    check("CpOL-AsHs-Coarse-BGUP-Cache", "coarse_bgup_cache_tools")
    assert_preserves_online_boundary_and_async_nodes(
        "CpOL-AsHs-Coarse", "CpOL-AsHs-Coarse-BGUP-Cache", "coarse"
    )


def test_spatial_cache_keeps_env_channels_separate():
    for directory, name in (
            ("cpol-ashs-new-BGUP-Cache", "new_channel_cache_tools"),
            ("CpOL-AsHs-Coarse-BGUP-Cache", "coarse_channel_cache_tools")):
        tools = load_tools(directory, name)
        env = np.zeros((2, 2, 2), dtype=int)
        env[0, 1, 1] = 1
        env[1, 0, 1] = 1
        agents = np.array([
            [0, 0, 1, 1, 120, 1, 1, 0, 0],
            [0, 0, 1, 1, 120, 2, 1, 0, 10],
        ], dtype=float)
        graph = nx.Graph()
        graph.add_nodes_from([
            ("worker0|0,0", "None", "None", "None", "None"),
            ("None", "car1|0,0", "None", "None", "None"),
        ])
        calls = []
        original = tools.calculate_average_distance

        def counted(matrix, row, col):
            calls.append(matrix.ctypes.data)
            return original(matrix, row, col)

        tools.calculate_average_distance = counted
        try:
            tools.creatWeight(
                graph, env, agents, np.ones((2, 2)), 1, 10
            )
        finally:
            tools.calculate_average_distance = original

        assert env[0].ctypes.data in calls
        assert env[1].ctypes.data in calls


if __name__ == "__main__":
    test_new_bgup_cache()
    test_new_preserves_online_boundary_and_async_nodes()
    test_coarse_bgup_cache()
    test_spatial_cache_keeps_env_channels_separate()
    print("CpOL BGUP-Cache checks passed")

