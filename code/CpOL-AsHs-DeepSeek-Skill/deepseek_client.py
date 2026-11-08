"""DeepSeek official HTTPS chat/completions API, JSON mode (stdlib only)."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

SKILL = Path(__file__).parent / 'skills/mdc-scheduling/SKILL.md'


class DeepSeek:
    def __init__(self, model='deepseek-flash', select_max_tokens=4096, evolve_max_tokens=16384):
        self.key = os.environ.get('DEEPSEEK_API_KEY')
        if not self.key:
            raise ValueError('Set DEEPSEEK_API_KEY in your terminal; never commit the key.')
        self.model = model
        if any(type(value) is not int or value <= 0
               for value in (select_max_tokens, evolve_max_tokens)):
            raise ValueError('Token budgets must be positive integers')
        self.select_max_tokens = select_max_tokens
        self.evolve_max_tokens = evolve_max_tokens
        self.last_attempts = []

    def ask(self, payload):
        self.last_attempts = []
        mode = payload.get('mode')
        budget = self.evolve_max_tokens if mode == 'evolve' else self.select_max_tokens
        body = {'model': self.model, 'stream': False,
                'response_format': {'type': 'json_object'}, 'max_tokens': budget,
                'messages': [{'role': 'system', 'content': SKILL.read_text()},
                             {'role': 'user', 'content': json.dumps(payload, allow_nan=False)}]}
        for attempt in range(2):
            request = Request('https://api.deepseek.com/chat/completions',
                              data=json.dumps(body).encode(),
                              headers={'Authorization': 'Bearer ' + self.key,
                                       'Content-Type': 'application/json'})
            with urlopen(request, timeout=90) as response:
                data = json.load(response)
            choice = data['choices'][0]
            finish = choice['finish_reason']
            usage = data.get('usage', {})
            self.last_attempts.append({'mode': mode, 'attempt': attempt + 1,
                                       'max_tokens': body['max_tokens'],
                                       'finish_reason': finish, 'usage': usage})
            if mode == 'evolve' or finish == 'length':
                print(f"DeepSeek {mode}：finish_reason={finish}，"
                      f"max_tokens={body['max_tokens']}，usage={json.dumps(usage)}", flush=True)
            if finish == 'length' and attempt == 0:
                body['max_tokens'] *= 2
                print(f"DeepSeek 响应被截断：将生成上限提高到 {body['max_tokens']}，重试一次。", flush=True)
                continue
            if finish != 'stop':
                raise ValueError(f'DeepSeek response incomplete: {finish}; '
                                 f'max_tokens={body["max_tokens"]}; usage={json.dumps(usage)}')
            return json.loads(choice['message']['content'])
