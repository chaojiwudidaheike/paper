import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from policy import VARIABLES, default_policy, expression, rank, validate
from runner import choose, evaluate_proposal, main
from deepseek_client import DeepSeek


class SchedulingTests(unittest.TestCase):
    def test_two_episodes_through_http_adapter_and_real_simulator(self):
        """Only external HTTP is substituted; formulas, simulation, gate and disk are real."""
        payloads = []
        proposal = default_policy()
        proposal['formulas']['task'] = 'task / (1 + mean_duration)'
        proposal['weights'].update(task=.30, pair=.45)

        class Response:
            def __init__(self, content): self.content = content
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'choices':[{'finish_reason':'stop',
                    'message':{'content':json.dumps(self.content)}}]}).encode()

        def transport(request, timeout):
            body = json.loads(request.data)
            self.assertIn('mdc-scheduling', body['messages'][0]['content'])
            payload = json.loads(body['messages'][1]['content'])
            payloads.append(payload)
            if payload['mode'] == 'select':
                return Response({'position':payload['candidates'][0]['position'], 'reason':'fixture'})
            return Response({'policy':proposal, 'reason':'duration-aware task potential'})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = np.zeros((2,2,2)); env[0,1,1] = env[1,1,1] = 1
            data = {'Env':env, 'Task_Time':np.ones((2,2)),
                    'Order_Finall':np.array([[0,0,1,1,100,1,1,0,0],
                                            [0,0,1,1,100,3,1,10,10],
                                            [0,0,1,1,100,2,1,0,2]], dtype=float)}
            savemat(root/'test.mat', data)
            argv = ['runner.py','--episodes','2','--scope','2','--time-limit','10',
                    '--dataset',str(root/'test.mat'),'--state',str(root/'state')]
            with patch('sys.argv', argv), patch.dict('os.environ', {'DEEPSEEK_API_KEY':'test-only'}), \
                 patch('deepseek_client.urlopen', transport):
                main()
            reports = [json.loads(p.read_text()) for p in (root/'state').glob('*-*.json')]
            self.assertEqual(len(reports),2)
            for report in reports:
                self.assertEqual(report['run']['completed'],1)
                self.assertEqual(report['run']['history'][0]['decision']['source'],'deepseek')
                self.assertEqual(report['proposal']['policy'],proposal)
                self.assertEqual(report['evolution_api_attempts'][0]['max_tokens'], 16384)
                self.assertEqual(report['evolution_api_attempts'][0]['finish_reason'], 'stop')
                self.assertFalse(report['evaluation']['accepted'])  # Equal count must not promote.
                old = report['evaluation']['baseline']['history'][0]['selected']['components']['task']
                new = report['evaluation']['candidate']['history'][0]['selected']['components']['task']
                self.assertLess(new,old)  # New formula actually executed during full replay.
            self.assertEqual(json.loads((root/'state/policy.json').read_text()), default_policy())
            self.assertEqual([p['mode'] for p in payloads],['select','evolve','select','evolve'])
            self.assertEqual(len(payloads[-1]['previous_attempts']),1)

    def test_formula_can_change_ranking(self):
        a, b = dict.fromkeys(VARIABLES, 0.), dict.fromkeys(VARIABLES, 0.)
        a.update(task=10, mean_duration=100)
        b.update(task=5, mean_duration=1)
        candidates = [{'position':[0,0], 'features':a}, {'position':[1,1], 'features':b}]
        old = default_policy()
        new = copy.deepcopy(old)
        new['formulas']['task'] = 'task / (1 + mean_duration)'
        validate(new)
        self.assertEqual(rank(candidates, old)[0]['position'], [0,0])
        self.assertEqual(rank(candidates, new)[0]['position'], [1,1])
        new['weights'].update(task=1., pair=0., charge=0., imbalance=0., move=0., energy_risk=0.)
        validate(new)

    def test_expression_rejects_executable_and_nonfinite_input(self):
        for source in ("__import__('os')", '(1).__class__', '1/0', '1e309', '2**100', '[task][0]'):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                expression(source, dict.fromkeys(VARIABLES, 1))

    def test_official_request_and_skill_loaded(self):
        captured = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'choices':[{'finish_reason':'stop',
                    'message':{'content':'{"position":[0,0],"reason":"test"}'}}]}).encode()
        def request(req, timeout):
            captured.append(req)
            return Response()
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY':'test-not-real'}), patch('deepseek_client.urlopen', request):
            client = DeepSeek()
            ranked = [{'position':[0,0]}]
            self.assertEqual(choose(ranked, default_policy(), client, [], 1)[1]['source'], 'deepseek')
        body = json.loads(captured[0].data)
        self.assertEqual(captured[0].full_url, 'https://api.deepseek.com/chat/completions')
        self.assertIn('mdc-scheduling', body['messages'][0]['content'])
        self.assertEqual(body['response_format'], {'type':'json_object'})

    def test_invalid_model_position_rejected(self):
        class Client:
            def ask(self, payload): return {'position':[99,99]}
        with self.assertRaises(ValueError):
            choose([{'position':[0,0]}], default_policy(), Client(), [], 1)

    def test_full_replay_keeps_equal_policy_and_input(self):
        env = np.zeros((2,2,2)); env[0,1,1] = env[1,1,1] = 1
        data = {'Env':env, 'Task_Time':np.ones((2,2)),
                'Order_Finall':np.array([[0,0,1,1,100,1,1,0,0],
                                        [0,0,1,1,100,3,1,10,10],
                                        [0,0,1,1,100,2,1,0,2]], dtype=float)}
        snapshot = copy.deepcopy(data)
        accepted, old, new = evaluate_proposal(data, default_policy(), default_policy(), 2, 10, 10)
        self.assertFalse(accepted)
        self.assertEqual(old['completed'], 1)
        self.assertEqual(new['completed'], 1)
        for key in data: np.testing.assert_array_equal(data[key], snapshot[key])

    def test_gate_accepts_only_strict_improvement(self):
        with patch('runner.simulate', side_effect=[{'completed':2},{'completed':3}]):
            self.assertTrue(evaluate_proposal({}, default_policy(), default_policy(), 2,10,10)[0])
        with patch('runner.simulate', side_effect=[{'completed':3},{'completed':2}]):
            self.assertFalse(evaluate_proposal({}, default_policy(), default_policy(), 2,10,10)[0])

    def test_accepted_policy_saved_and_loaded_next_episode(self):
        proposed = default_policy()
        proposed['formulas']['task'] = 'task / (1 + mean_duration)'
        class Client:
            def ask(self, payload):
                return {'policy': proposed, 'reason':'Prefer shorter tasks'}
        result = {'completed':3, 'total':4, 'rate':.75, 'history':[]}
        with tempfile.TemporaryDirectory() as temp:
            argv = ['runner.py','--episodes','2','--state',temp]
            with patch('sys.argv', argv), patch('runner.DeepSeek', return_value=Client()), \
                 patch('runner.loadmat', return_value={}), \
                 patch('runner.simulate', return_value=result) as run, \
                 patch('runner.evaluate_proposal', return_value=(True, {'completed':2, 'total':4, 'rate':.5}, result)):
                main()
            saved = json.loads((Path(temp)/'policy.json').read_text())
            self.assertEqual(saved, proposed)
            self.assertEqual(run.call_args_list[1].args[1], proposed)
            reports = list(Path(temp).glob('*-*.json'))
            self.assertEqual(len(reports), 2)
            self.assertTrue(json.loads(reports[0].read_text())['evaluation']['accepted'])


if __name__ == '__main__':
    unittest.main()
