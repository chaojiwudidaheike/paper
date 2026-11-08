import io
import json
import unittest
from unittest.mock import patch

from deepseek_client import DeepSeek


class ResponseTests(unittest.TestCase):
    @staticmethod
    def response(reason, content, tokens=10):
        return io.BytesIO(json.dumps({
            'choices': [{'finish_reason': reason, 'message': {'content': content}}],
            'usage': {'completion_tokens': tokens},
        }).encode())

    def test_evolve_retries_truncation_with_larger_budget(self):
        responses = [self.response('length', '{', 16384),
                     self.response('stop', '{"policy": {}}')]
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'}), \
             patch('deepseek_client.urlopen', side_effect=responses) as send:
            client = DeepSeek()
            self.assertEqual(client.ask({'mode': 'evolve'}), {'policy': {}})
        bodies = [json.loads(call.args[0].data) for call in send.call_args_list]
        self.assertEqual([b['max_tokens'] for b in bodies], [16384, 32768])
        self.assertEqual(bodies[0]['messages'], bodies[1]['messages'])
        self.assertEqual(client.last_attempts[0]['usage']['completion_tokens'], 16384)
        self.assertEqual(client.last_attempts[1]['finish_reason'], 'stop')

    def test_repeated_length_stops_after_one_retry(self):
        # Even parseable JSON must be rejected if the provider reports truncation.
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'}), \
             patch('deepseek_client.urlopen', side_effect=[
                 self.response('length', '{}'), self.response('length', '{}')]) as send:
            client = DeepSeek(evolve_max_tokens=8000)
            with self.assertRaisesRegex(ValueError, 'length'):
                client.ask({'mode': 'evolve'})
        self.assertEqual(send.call_count, 2)
        self.assertEqual(client.last_attempts[-1]['max_tokens'], 16000)

    def test_select_budget_and_other_finish_reasons(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'}), \
             patch('deepseek_client.urlopen', side_effect=[
                 self.response('stop', '{"position": [0, 0]}'),
                 self.response('content_filter', '{}')]) as send:
            client = DeepSeek(select_max_tokens=2000)
            client.ask({'mode': 'select'})
            self.assertEqual(json.loads(send.call_args.args[0].data)['max_tokens'], 2000)
            with self.assertRaisesRegex(ValueError, 'content_filter'):
                client.ask({'mode': 'evolve'})
        self.assertEqual(send.call_count, 2)
        self.assertEqual(len(client.last_attempts), 1)


if __name__ == '__main__':
    unittest.main()
