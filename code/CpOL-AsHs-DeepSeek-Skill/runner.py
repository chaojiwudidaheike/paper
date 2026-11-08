"""uv run --project MDC python MDC/code/CpOL-AsHs-DeepSeek-Skill/runner.py"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from scipy.io import loadmat
import coarse_scheduler as coarse
from Tools import build_graph, creatWeight, updateEnv
from algorithm import max_weight_max_independent_set
from deepseek_client import DeepSeek
from policy import default_policy, rank, validate

HERE = Path(__file__).resolve().parent


def candidates(env, agents, durations, scope, step, interval, limit, previous):
    if not 1 <= scope <= min(env.shape[1:]):
        raise ValueError('Invalid coverage scope')
    tasks = list(zip(*np.nonzero(env[1] == 1)))
    chargers = list(zip(*np.nonzero(env[0] == 1)))
    result = []
    for row in range(env.shape[1] - scope + 1):
        for col in range(env.shape[2] - scope + 1):
            features = coarse._region_features(env, agents, durations, row, col,
                                               scope, step, interval, limit, previous, set())
            controlled = coarse._controlled_agents(agents, row, col, scope, step*interval, set())
            triples = coarse._task_triples(env, agents, durations, controlled, tasks, chargers,
                                           (step-1)*interval, step*interval, limit, set())
            features.update(workers=len(controlled[1]), cars=len(controlled[2]),
                            uavs=len(controlled[3]), remaining=len(tasks),
                            time_left=limit-(step-1)*interval, triples=len(triples),
                            mean_duration=sum(t['finish']-(step-1)*interval for t in triples)/max(1,len(triples)),
                            mean_energy=sum(t['energy_ratio'] for t in triples)/max(1,len(triples)),
                            mean_slack=sum(t['slack'] for t in triples)/max(1,len(triples)))
            features = {k: float(v) for k, v in features.items() if k not in ('row', 'lin')}
            result.append({'position': [row, col], 'features': features})
    return result


def choose(ranked, policy, client, history, step):
    best = ranked[0]
    if client is None:
        return best, {'source': 'offline-arithmetic', 'position': best['position']}
    response = client.ask({'mode': 'select', 'step': step, 'policy': policy,
                           'candidates': ranked, 'feedback': history[-3:]})
    if not isinstance(response, dict):
        raise ValueError('Model decision must be a JSON object')
    position = response.get('position')
    if (not isinstance(position, list) or len(position) != 2
            or any(type(v) is not int for v in position) or position != best['position']):
        raise ValueError('Model position must equal the highest-scoring candidate')
    return best, {**response, 'source': 'deepseek'}


def simulate(data, policy, scope, limit, interval, client=None, verbose=False):
    random.seed(0)
    np.random.seed(0)
    env, agents, durations = (data[k].copy() for k in ('Env','Order_Finall','Task_Time'))
    total = int(env[1].sum())
    previous = None
    history = []
    started = time.monotonic()
    for step in range(1, limit//interval+1):
        ranked = rank(candidates(env, agents, durations, scope, step, interval, limit, previous), policy)
        best, response = choose(ranked, policy, client, history, step)
        row, col = best['position']
        # Keep the source BGUP/caching, weights, MWIS, sorting and transition logic.
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_graph(env, agents, durations, row, col, scope, step, interval, limit)
            graph = creatWeight(graph, env, agents, durations, step, interval)
            selected, _ = max_weight_max_independent_set(graph)
            selected = sorted(selected, key=lambda x: (graph.nodes[x]['weight'], x), reverse=True)
            updated = updateEnv(env, agents, durations, selected, step, interval)
        env, agents, durations = updated[:3]
        record = {'step': step, 'completed': total-int(env[1].sum()),
                  'decision': response, 'selected': best}
        history.append(record)
        previous = (row,col)
        if verbose:
            print(f"step={step} position={previous} completed={record['completed']}/{total}", flush=True)
    return {'completed': total-int(env[1].sum()), 'total': total,
            'rate': (total-int(env[1].sum()))/total if total else 0,
            'seconds': time.monotonic()-started, 'history': history}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = output.name
    os.replace(temporary, path)


def evaluate_proposal(data, current, proposal, scope, limit, interval):
    validate(proposal)
    baseline = simulate(data, current, scope, limit, interval)
    candidate = simulate(data, proposal, scope, limit, interval)
    accepted = candidate['completed'] > baseline['completed']
    return accepted, baseline, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=HERE.parents[1]/'data/Mat/Random_1.mat')
    parser.add_argument('--scope', type=int, default=20)
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--model', default=os.environ.get('DEEPSEEK_MODEL','deepseek-flash'))
    parser.add_argument('--select-max-tokens', type=int, default=4096,
                        help='Initial output token budget for position selection; doubled once on length')
    parser.add_argument('--evolve-max-tokens', type=int, default=16384,
                        help='Initial output token budget for policy proposals; doubled once on length')
    parser.add_argument('--offline', action='store_true', help='Arithmetic verification only; no model or evolution')
    parser.add_argument('--state', type=Path, default=HERE/'state')
    args = parser.parse_args()
    if args.episodes < 1 or args.interval <= 0 or args.time_limit <= 0 or args.time_limit % args.interval:
        parser.error('Positive episodes/time/interval required; time must be divisible by interval')
    if args.select_max_tokens <= 0 or args.evolve_max_tokens <= 0:
        parser.error('Token budgets must be positive')
    client = None if args.offline else DeepSeek(args.model, args.select_max_tokens, args.evolve_max_tokens)
    data = loadmat(args.dataset, mat_dtype=True)
    policy_path = args.state/'policy.json'
    current = validate(json.loads(policy_path.read_text()) if policy_path.exists() else default_policy())
    session = time.time_ns()
    feedback = []
    archive = args.state/'last_feedback.json'
    if archive.exists():
        feedback = json.loads(archive.read_text())[-5:]
    for episode in range(1, args.episodes+1):
        print(f"\n第 {episode}/{args.episodes} 轮：开始正式仿真（使用本轮开始时的策略）", flush=True)
        report = {'dataset': str(args.dataset.resolve()), 'scope': args.scope,
                  'time_limit': args.time_limit, 'interval': args.interval,
                  'model': args.model if client else None, 'policy': current,
                  'select_max_tokens': args.select_max_tokens,
                  'evolve_max_tokens': args.evolve_max_tokens,
                  'episode': episode, 'session': session}
        report['run'] = simulate(data, current, args.scope, args.time_limit, args.interval, client, True)
        print(f"本轮正式仿真：完成 {report['run']['completed']}/{report['run']['total']} 个任务"
              f"（{report['run']['rate']:.2%}）", flush=True)
        destination = args.state/f'{session}-{episode}.json'
        save(destination, report)
        if client:
            try:
                print('策略更新：正在请求 DeepSeek 提出新公式和权重……', flush=True)
                proposal = client.ask({'mode': 'evolve', 'policy': current,
                                       'episode': report['run'], 'previous_attempts': feedback[-5:]})
                if not isinstance(proposal, dict):
                    raise ValueError('Model proposal must be a JSON object')
                report['proposal'] = proposal
                print('策略回放评估：新旧策略将分别从相同初始数据重新仿真，不计入本轮正式仿真结果。', flush=True)
                accepted, baseline, candidate = evaluate_proposal(
                    data, current, proposal['policy'], args.scope, args.time_limit, args.interval)
                report['evaluation'] = {'accepted': accepted, 'baseline': baseline, 'candidate': candidate}
                if accepted:
                    current = proposal['policy']
                    save(policy_path, current)
                feedback.append({'proposal': proposal, 'accepted': accepted,
                                 'baseline_completed': baseline['completed'],
                                 'candidate_completed': candidate['completed']})
                print(f"策略回放评估：旧策略 {baseline['completed']}/{baseline['total']}"
                      f"（{baseline['rate']:.2%}），新策略 {candidate['completed']}/{candidate['total']}"
                      f"（{candidate['rate']:.2%}）", flush=True)
                if accepted:
                    next_run = ('下一轮正式仿真' if episode < args.episodes
                                else '下次使用同一 state 目录启动的仿真')
                    print(f'新策略已接受并保存，将用于{next_run}。', flush=True)
                else:
                    reason = ('完成任务数持平' if candidate['completed'] == baseline['completed']
                              else '完成任务数下降')
                    print(f'新策略未接受：{reason}，保留旧策略。', flush=True)
            except (ValueError, KeyError, TypeError, ArithmeticError, SyntaxError) as error:
                report['evolution_error'] = str(error)
                feedback.append({'proposal': report.get('proposal'), 'error': str(error)})
                print(f'策略更新失败（非效果判定）：{error}。保留旧策略，本轮正式仿真结果不变。', flush=True)
            finally:
                report['evolution_api_attempts'] = getattr(client, 'last_attempts', [])
                save(destination, report)
                save(archive, feedback[-5:])
        if not policy_path.exists():
            save(policy_path, current)


if __name__ == '__main__':
    main()
