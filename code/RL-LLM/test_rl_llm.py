import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_encoder import GNNEncoder, NODE_FEATURE_DIM, extract_node_features
from llm_encoder import SimpleTextEncoder
from pruning_agent import PruningPPO
from pruning_env import PruningEnv
from Tools import build_graph
from runner import (call_silently, parse_args, print_evaluation_table,
                    resolve_center_scope, summarize_evaluation, write_output)


class RLLLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def sample_agents():
        # date, start, row, col, end, type, speed, power, full_power
        return np.asarray([
            [0, 0, 2, 2, 120, 1, 1, 0, 0],
            [0, 0, 3, 3, 120, 3, 2, 8, 10],
        ], dtype=float)

    def test_features_are_finite_and_flip_invariant(self):
        env = np.zeros((2, 5, 5), dtype=float)
        env[1, 3, 3] = 1
        task_time = np.zeros((5, 5), dtype=float)
        task_time[3, 3] = 4
        node = ('worker0|1,1', 'None', 'uav1|2,2', 'task3,3', 'None')
        graph = nx.Graph()
        graph.add_node(node, weight=1.25)
        feat, _ = extract_node_features(graph, env, self.sample_agents(), task_time)
        self.assertEqual(feat.shape, (1, NODE_FEATURE_DIM))
        self.assertTrue(np.isfinite(feat).all())

        flipped_env = env[:, ::-1, :].copy()
        flipped_time = task_time[::-1, :].copy()
        flipped_agents = self.sample_agents()
        flipped_agents[:, 2] = 6 - flipped_agents[:, 2]
        flipped_node = ('worker0|3,1', 'None', 'uav1|2,2', 'task1,3', 'None')
        flipped_graph = nx.Graph()
        flipped_graph.add_node(flipped_node, weight=1.25)
        flipped_feat, _ = extract_node_features(
            flipped_graph, flipped_env, flipped_agents, flipped_time,
        )
        np.testing.assert_allclose(feat, flipped_feat, atol=1e-6)

        permuted_agents = self.sample_agents()[[1, 0]].copy()
        permuted_node = ('worker1|1,1', 'None', 'uav0|2,2', 'task3,3', 'None')
        permuted_graph = nx.Graph()
        permuted_graph.add_node(permuted_node, weight=1.25)
        permuted_feat, _ = extract_node_features(
            permuted_graph, env, permuted_agents, task_time,
        )
        np.testing.assert_allclose(feat, permuted_feat, atol=1e-6)

        empty, mapping = extract_node_features(nx.Graph(), env, self.sample_agents())
        self.assertEqual(empty.shape, (0, NODE_FEATURE_DIM))
        self.assertEqual(mapping, {})

    def test_safe_pruning_keeps_greedy_and_group_anchors(self):
        nodes = [
            ('worker0|0,0', 'None', 'uav2|0,1', 'task1,1', 'None'),
            ('worker1|1,0', 'None', 'uav3|1,1', 'task1,1', 'None'),
            ('worker0|0,0', 'None', 'None', 'task2,2', 'None'),
        ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for index, node in enumerate(nodes):
            graph.nodes[node]['weight'] = 3 - index
        graph.add_edge(nodes[0], nodes[1])
        llm = SimpleTextEncoder(embed_dim=8, num_heads=2, num_layers=1, max_len=32)
        gnn = GNNEncoder(in_dim=NODE_FEATURE_DIM, hidden_dim=4, out_dim=8,
                         num_heads=2, dropout=0)
        env = PruningEnv(llm, gnn)
        node_to_idx = {node: index for index, node in enumerate(nodes)}
        probs = torch.tensor([[0.2, 0.9, 0.1]])
        mask, info = env.safe_keep_mask(
            graph, probs, torch.ones_like(probs), node_to_idx, keep_ratio=0.34,
        )
        kept = {node for node, index in node_to_idx.items() if mask[0, index] == 1}
        self.assertTrue(info['greedy_nodes'].issubset(kept))
        self.assertIn(nodes[1], kept)  # best-scoring candidate for task1,1
        self.assertIn(nodes[2], kept)  # only candidate for task2,2

    def test_ppo_uses_stored_log_probs_and_checkpoint_roundtrip(self):
        agent = PruningPPO(embed_dim=8, update_epochs=2)
        for index, done in enumerate((False, True)):
            z = torch.randn(1, 8)
            node_count = 3 - index
            emb = torch.randn(1, node_count, 8)
            mask = torch.ones(1, node_count)
            action, probs, logs, fused, gate = agent.act(z, z, emb, mask)
            agent.store_transition(z, z, emb, mask, action, logs, 1.0, done)
        saved_logs = [item['old_log_probs'].copy() for item in agent.buffer]
        loss, _ = agent.update()
        self.assertTrue(np.isfinite(loss))
        for before, item in zip(saved_logs, agent.buffer):
            np.testing.assert_array_equal(before, item['old_log_probs'])

        llm = SimpleTextEncoder(embed_dim=8, num_heads=2, num_layers=1, max_len=32)
        gnn = GNNEncoder(in_dim=NODE_FEATURE_DIM, hidden_dim=4, out_dim=8,
                         num_heads=2, dropout=0)
        llm.eval(); gnn.eval(); agent.actor.eval(); agent.fusion.eval()
        text_before = llm(['state']).detach()
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'model')
            agent.save_model(prefix, llm, gnn)
            loaded_agent = PruningPPO(embed_dim=8, update_epochs=2)
            loaded_llm = SimpleTextEncoder(embed_dim=8, num_heads=2,
                                           num_layers=1, max_len=32)
            loaded_gnn = GNNEncoder(in_dim=NODE_FEATURE_DIM, hidden_dim=4,
                                    out_dim=8, num_heads=2, dropout=0)
            loaded_agent.load_model(prefix, loaded_llm, loaded_gnn)
            loaded_llm.eval()
            torch.testing.assert_close(text_before, loaded_llm(['state']).detach())
            for expected, actual in zip(agent.actor.parameters(), loaded_agent.actor.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_cli_verbosity_table_and_json_output(self):
        self.assertEqual(call_silently(True, lambda: (print('noise'), 7)[1]), 7)
        defaults = parse_args(['train', '--quiet'])
        self.assertTrue(defaults.quiet)
        self.assertIsNone(defaults.center_scope)
        self.assertEqual(defaults.max_imitation_grad_nodes, 3000)
        self.assertEqual(defaults.keep_ratio, 0.55)
        self.assertEqual(defaults.quality_threshold, 0.98)
        self.assertEqual(defaults.min_dataset_quality, 0.92)
        self.assertEqual(defaults.mean_quality_threshold, 0.96)
        self.assertTrue(parse_args(['eval', '--verbose']).verbose)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(['infer', '--quiet', '--verbose'])

        bicycle = np.zeros((2, 25, 16))
        didi = np.zeros((2, 10, 9))
        random_data = np.zeros((2, 30, 30))
        self.assertEqual(resolve_center_scope('BicycleOrder.mat', bicycle), 14)
        self.assertEqual(resolve_center_scope('DidiOrders.mat', didi), 8)
        self.assertEqual(resolve_center_scope('Random_1.mat', random_data), 20)
        self.assertEqual(resolve_center_scope('DidiOrders.mat', didi, 7), 7)
        with self.assertRaises(ValueError):
            resolve_center_scope('DidiOrders.mat', didi, 20)

        rows = [{'dataset': 'Random_1.mat', 'weight_ratio': 0.99,
                 'completion_ratio': 1.0, 'keep_ratio': 0.5,
                 'EE': 0.125, 'ATRT': 12.0, 'fallbacks': 0, 'passed': True}]
        output = StringIO()
        with redirect_stdout(output):
            print_evaluation_table(rows)
        rendered = output.getvalue()
        self.assertIn('数据集', rendered)
        self.assertIn('MWIS比例', rendered)
        self.assertIn('通过', rendered)

        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / 'nested' / 'result.json'
            args = parse_args(['infer', '--quiet', '--output', str(result_path)])
            with redirect_stdout(StringIO()):
                write_output(args, {'config': {'mode': 'infer'},
                                    'steps': [{'quality': float('inf')}],
                                    'summary': {'passed': True}})
            saved = result_path.read_text(encoding='utf-8')
            self.assertIn('"mode": "infer"', saved)
            self.assertIn('"quality": null', saved)

    def test_infeasible_nodes_are_not_reintroduced_by_edges(self):
        env = np.zeros((2, 2, 2), dtype=float)
        env[0, 0, 0] = 1
        env[1, 1, 1] = 1
        task_time = np.zeros((2, 2), dtype=float)
        task_time[1, 1] = 200
        agents = np.asarray([
            [0, 0, 1, 1, 120, 1, 1, 0, 0],
            [0, 0, 1, 1, 120, 3, 1, 100, 100],
        ], dtype=float)
        graph = call_silently(
            True, build_graph, env, agents, task_time,
            0, 0, 2, 1, 10, 120,
        )
        self.assertFalse(any(
            node[0] != 'None' and node[2] != 'None' and node[3] != 'None'
            for node in graph
        ))

    def test_layered_acceptance_thresholds(self):
        args = parse_args(['eval'])
        rows = [
            {'weight_ratio': 0.92, 'completion_ratio': 0.94,
             'keep_ratio': 0.54, 'passed': True},
            {'weight_ratio': 1.00, 'completion_ratio': 0.98,
             'keep_ratio': 0.56, 'passed': True},
        ]
        summary = summarize_evaluation(rows, args)
        self.assertTrue(summary['all_dataset_minimums_passed'])
        self.assertTrue(summary['mean_quality_passed'])
        self.assertAlmostEqual(summary['mean_keep_ratio'], 0.55)
        self.assertTrue(summary['overall_passed'])

        low_mean_rows = [
            {'weight_ratio': 0.94, 'completion_ratio': 0.94,
             'keep_ratio': 0.50, 'passed': True},
            {'weight_ratio': 0.94, 'completion_ratio': 0.94,
             'keep_ratio': 0.50, 'passed': True},
        ]
        self.assertFalse(
            summarize_evaluation(low_mean_rows, args)['all_quality_passed'])

        high_keep_rows = [dict(row, keep_ratio=0.56) for row in rows]
        high_keep = summarize_evaluation(high_keep_rows, args)
        self.assertFalse(high_keep['keep_ratio_passed'])
        self.assertFalse(high_keep['overall_passed'])


if __name__ == '__main__':
    unittest.main()
