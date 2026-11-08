"""Cross-dataset RL-LLM graph-pruning runner.

Flow: select region -> build graph -> create weight -> safe learned pruning
-> MWIS -> update environment.
"""

import argparse
import copy
import contextlib
import io
import json
import math
import random
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from Tools import build_graph, creatWeight, select_optimal_region, updateEnv
from algorithm import max_weight_max_independent_set
from gnn_encoder import NODE_FEATURE_DIM, create_gnn_encoder
from llm_encoder import create_llm_encoder
from pruning_agent import create_pruning_agent
from pruning_env import PruningEnv


HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / '../../data/Mat').resolve()
DEFAULT_MODEL = HERE / 'model/rl_llm'
DEFAULT_CENTER_SCOPE = 20
DATASET_CENTER_SCOPES = {
    'BicycleOrder.mat': 14,
    'DidiOrders.mat': 8,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval', 'infer'))
    parser.add_argument('--data', default='Random_1.mat', help='Dataset name/path for infer')
    parser.add_argument('--model-prefix', default=str(DEFAULT_MODEL))
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--imitation-epochs', type=int, default=1)
    parser.add_argument('--imitation-steps', type=int, default=12)
    parser.add_argument('--max-imitation-grad-nodes', type=int, default=3000,
                        help='模仿阶段允许LLM/GNN反向传播的最大图节点数')
    parser.add_argument('--keep-ratio', type=float, default=0.55)
    parser.add_argument('--quality-threshold', type=float, default=0.98,
                        help='训练奖励的质量目标')
    parser.add_argument('--min-dataset-quality', type=float, default=0.92,
                        help='单数据集MWIS与完成率最低验收值')
    parser.add_argument('--mean-quality-threshold', type=float, default=0.96,
                        help='全数据集平均MWIS与完成率最低验收值')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--center-scope', type=int, default=None,
                        help='MDC窗口边长；默认按数据集自动选择，传值则全局覆盖')
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--single-step', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Testing aid; defaults to the full horizon')
    parser.add_argument('--no-augment', action='store_true')
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', action='store_true', help='仅显示最终摘要')
    verbosity.add_argument('--verbose', action='store_true', help='显示底层算法调试信息')
    parser.add_argument('--output', help='将结构化运行结果写入JSON文件')
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_paths():
    return sorted(DATA_DIR.glob('*.mat'))


def resolve_data(value):
    path = Path(value)
    if not path.exists():
        path = DATA_DIR / value
    if not path.exists():
        raise FileNotFoundError(f'dataset not found: {value}')
    return path.resolve()


def load_dataset(path):
    data = loadmat(path, mat_dtype=True)
    return np.copy(data['Env']), np.copy(data['Order_Finall']), np.copy(data['Task_Time'])


def resolve_center_scope(path, env, override=None):
    scope = (override if override is not None else
             DATASET_CENTER_SCOPES.get(Path(path).name, DEFAULT_CENTER_SCOPE))
    max_scope = min(env.shape[1:])
    if not 1 <= scope <= max_scope:
        raise ValueError(
            f'{Path(path).name}: center scope must be in [1, {max_scope}], got {scope}'
        )
    return scope


def augment_dataset(env_data, rng):
    """Apply semantics-preserving flips and agent-row permutation."""
    env, agents, task_time = (np.copy(x) for x in env_data)
    if rng.random() < 0.5:
        env = env[:, ::-1, :].copy()
        task_time = task_time[::-1, :].copy()
        agents[:, 2] = env.shape[1] + 1 - agents[:, 2]
    if rng.random() < 0.5:
        env = env[:, :, ::-1].copy()
        task_time = task_time[:, ::-1].copy()
        agents[:, 3] = env.shape[2] + 1 - agents[:, 3]
    agents = agents[rng.permutation(len(agents))].copy()
    return env, agents, task_time


def create_components(device):
    llm = create_llm_encoder(embed_dim=128, use_pretrained=False).to(device)
    gnn = create_gnn_encoder(in_dim=NODE_FEATURE_DIM, hidden_dim=64,
                             out_dim=128, num_heads=4, dropout=0.1).to(device)
    agent = create_pruning_agent(embed_dim=128, device=device)
    env = PruningEnv(llm, gnn, device=device)
    return llm, gnn, agent, env


def solve_graph(graph):
    if not graph.nodes:
        return set(), 0.0
    independent_set, weight = max_weight_max_independent_set(graph)
    return independent_set, float(weight)


def relative_quality(value, baseline):
    if baseline <= 1e-8:
        return 1.0 if value <= 1e-8 else float('inf')
    return value / baseline


def call_silently(enabled, function, *args, **kwargs):
    if not enabled:
        return function(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def emit(args, message='', final=False):
    if final or not args.quiet:
        print(message)


def print_run_header(args, device, paths):
    if args.quiet:
        return
    datasets = args.data if args.mode == 'infer' else f'{len(paths)}个数据集'
    print('=' * 72)
    print(f'RL-LLM | 模式={args.mode.upper()} | 设备={device}')
    print(f'数据={datasets} | 模型={args.model_prefix}')
    scope = args.center_scope if args.center_scope is not None else '自动'
    print(f'MDC窗口={scope}')
    print(f'随机种子={args.seed} | 保留率≤{args.keep_ratio:.0%} | '
          f'训练目标={args.quality_threshold:.0%}')
    print(f'验收门槛: 单数据集≥{args.min_dataset_quality:.0%} | '
          f'全数据集平均≥{args.mean_quality_threshold:.0%}')
    print('=' * 72)


def format_table(headers, rows, aligns=None):
    text_rows = [[str(value) for value in row] for row in rows]
    display_width = lambda value: sum(
        2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        for char in str(value)
    )
    widths = [display_width(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    aligns = aligns or ['<'] * len(headers)

    def render(row):
        cells = []
        for index, value in enumerate(row):
            padding = ' ' * (widths[index] - display_width(value))
            cells.append(padding + value if aligns[index] == '>' else value + padding)
        return '  '.join(cells)

    separator = '  '.join('-' * width for width in widths)
    return '\n'.join([render([str(x) for x in headers]), separator,
                      *(render(row) for row in text_rows)])


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_output(args, payload):
    if not args.output:
        return
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
                    encoding='utf-8')
    emit(args, f'结果已保存: {path}', final=True)


def run_config(args, device):
    return {
        'mode': args.mode, 'device': str(device), 'model_prefix': args.model_prefix,
        'data': args.data, 'episodes': args.episodes, 'seed': args.seed,
        'max_imitation_grad_nodes': args.max_imitation_grad_nodes,
        'keep_ratio': args.keep_ratio, 'quality_threshold': args.quality_threshold,
        'min_dataset_quality': args.min_dataset_quality,
        'mean_quality_threshold': args.mean_quality_threshold,
        'center_scope': args.center_scope if args.center_scope is not None else 'auto',
        'center_scope_defaults': {**DATASET_CENTER_SCOPES,
                                  '*': DEFAULT_CENTER_SCOPE},
        'time_limit': args.time_limit,
        'single_step': args.single_step,
    }


def run_episode(env_data, components, args, *, center_scope, pruning, collect=False,
                compare_full=False, deterministic=True, quiet=None):
    llm, gnn, agent, pruning_env = components
    env, agents, task_time = (np.copy(x) for x in env_data)
    quiet = args.quiet if quiet is None else quiet
    timesteps = args.max_steps or int(args.time_limit / args.single_step)
    total_tasks = int(np.sum(env[1]))
    result = np.zeros(timesteps)
    total_uav_task = total_uav_charge = 0.0
    total_charges = 0
    responses = []
    transitions = []
    step_records = []
    full_weight_total = pruned_weight_total = 0.0
    original_nodes = kept_nodes = fallback_count = budget_exceeded_count = 0
    timings = {'build': 0.0, 'weight': 0.0, 'reference_mwis': 0.0,
               'prune': 0.0, 'mwis': 0.0}
    silent_internal = not args.verbose

    for step in range(1, timesteps + 1):
        step_started = time.perf_counter()
        timing_before = dict(timings)
        did_fallback = False
        center = select_optimal_region(env, agents, center_scope,
                                       step, args.single_step)
        started = time.perf_counter()
        graph = call_silently(
            silent_internal, build_graph, env, agents, task_time, *center,
            center_scope, step, args.single_step, args.time_limit,
        )
        timings['build'] += time.perf_counter() - started
        started = time.perf_counter()
        graph = creatWeight(graph, env, agents, task_time, step, args.single_step)
        timings['weight'] += time.perf_counter() - started
        n_original = len(graph)
        original_nodes += n_original

        started = time.perf_counter()
        full_set, full_weight = (call_silently(silent_internal, solve_graph, graph)
                                 if compare_full else (set(), 0.0))
        timings['reference_mwis'] += time.perf_counter() - started
        full_weight_total += full_weight
        chosen_graph = graph
        safety = None
        transition = None

        if pruning and n_original:
            started = time.perf_counter()
            state = pruning_env.extract_state(
                graph, agents, env, task_time, *center, center_scope,
                step, args.single_step, args.time_limit,
            )
            z_llm, z_gnn, node_emb, node_mask, node_to_idx = state
            sampled, probs, sampled_log_probs, h_fused, gate = agent.act(
                z_llm, z_gnn, node_emb, node_mask,
                deterministic=deterministic, min_keep_ratio=args.keep_ratio,
            )
            keep_mask, safety = pruning_env.safe_keep_mask(
                graph, probs, node_mask, node_to_idx, args.keep_ratio,
            )
            budget_exceeded_count += int(safety['budget_exceeded'])
            with torch.no_grad():
                old_log_probs, _ = agent.actor.evaluate_actions(
                    h_fused, node_emb, node_mask, keep_mask,
                )
            candidate = copy.deepcopy(graph)
            candidate, _ = pruning_env.apply_pruning(candidate, keep_mask, node_to_idx)
            if pruning_env.validate_pruning(candidate, safety):
                chosen_graph = candidate
            else:
                fallback_count += 1
                did_fallback = True
                chosen_graph = graph
                keep_mask = torch.ones_like(node_mask)
                with torch.no_grad():
                    old_log_probs, _ = agent.actor.evaluate_actions(
                        h_fused, node_emb, node_mask, keep_mask,
                    )
            kept_nodes += len(chosen_graph)
            timings['prune'] += time.perf_counter() - started
            transition = dict(z_llm=z_llm, z_gnn=z_gnn, node_emb=node_emb,
                              node_mask=node_mask, keep_mask=keep_mask,
                              old_log_probs=old_log_probs)
        else:
            kept_nodes += n_original

        started = time.perf_counter()
        selected, selected_weight = call_silently(silent_internal, solve_graph, chosen_graph)
        timings['mwis'] += time.perf_counter() - started
        pruned_weight_total += selected_weight
        before_completed = total_tasks - int(np.sum(env[1]))
        before_distance = total_uav_task
        update_result = call_silently(
            silent_internal, updateEnv, env, agents, task_time,
            sorted(selected, key=lambda n: chosen_graph.nodes[n]['weight'], reverse=True),
            step, args.single_step,
        )
        (env, agents, task_time, uav_task, uav_charge, charges,
         task_responses, worker_dist, car_dist) = update_result
        total_uav_task += uav_task
        total_uav_charge += uav_charge
        total_charges += charges
        responses.extend(task_responses)
        completed = total_tasks - int(np.sum(env[1]))
        result[step - 1] = completed

        if transition is not None and collect:
            quality = selected_weight / max(full_weight, 1e-8) if full_weight > 0 else 1.0
            keep = len(chosen_graph) / max(n_original, 1)
            step_completed = completed - before_completed
            step_distance = total_uav_task - before_distance
            efficiency = step_completed / max(step_distance, 1.0)
            transition['reward'] = (quality
                                    - 5.0 * max(0.0, args.quality_threshold - quality)
                                    - 0.1 * keep + 0.05 * efficiency)
            transition['done'] = step == timesteps
            transitions.append(transition)

        ratio = relative_quality(selected_weight, full_weight) if compare_full else 1.0
        step_timing = {name: timings[name] - timing_before[name] for name in timings}
        step_record = {
            'step': step, 'center': list(center), 'original_nodes': n_original,
            'kept_nodes': len(chosen_graph), 'weight': selected_weight,
            'quality_ratio': ratio,
            'budget_exceeded': bool(safety and safety['budget_exceeded']),
            'fallback': did_fallback, 'completed': completed,
            'timings': step_timing, 'elapsed': time.perf_counter() - step_started,
        }
        step_records.append(step_record)
        if not quiet:
            flags = []
            if step_record['budget_exceeded']:
                flags.append('超预算')
            if did_fallback:
                flags.append('回退')
            flag_text = '/'.join(flags) if flags else '正常'
            print(f'[时间步 {step:02d}/{timesteps:02d}] MDC=({center[0]},{center[1]}) '
                  f'节点={n_original}->{len(chosen_graph)} MWIS={selected_weight:.4f} '
                  f'质量={ratio:.1%} 状态={flag_text} 耗时={step_record["elapsed"]:.2f}s')

    completed = int(result[-1]) if len(result) else 0
    metrics = {
        'completed': completed,
        'total_tasks': total_tasks,
        'EE': completed / total_uav_task if total_uav_task else 0.0,
        'ATRT': sum(responses) / completed if completed else 0.0,
        'charge_count': total_charges,
        'uav_task_distance': total_uav_task,
        'uav_charge_distance': total_uav_charge,
        'weight_total': pruned_weight_total,
        'full_weight_total': full_weight_total,
        'weight_ratio': pruned_weight_total / max(full_weight_total, 1e-8)
                        if compare_full and full_weight_total else 1.0,
        'keep_ratio': kept_nodes / max(original_nodes, 1),
        'original_nodes': original_nodes,
        'kept_nodes': kept_nodes,
        'fallback_count': fallback_count,
        'budget_exceeded_count': budget_exceeded_count,
        'timings': timings,
        'steps': step_records,
    }
    return metrics, transitions


def imitation_pretrain(paths, components, args):
    llm, gnn, agent, pruning_env = components
    params = list(llm.parameters()) + list(gnn.parameters()) + \
             list(agent.fusion.parameters()) + list(agent.actor.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    rng = np.random.default_rng(args.seed)
    llm.train(); gnn.train(); agent.fusion.train(); agent.actor.train()
    losses = []
    history = []
    for epoch in range(args.imitation_epochs):
        epoch_started = time.perf_counter()
        epoch_losses = []
        samples = 0
        encoder_frozen_samples = 0
        shuffled = list(paths)
        random.shuffle(shuffled)
        for path in shuffled:
            env_data = load_dataset(path)
            center_scope = resolve_center_scope(path, env_data[0], args.center_scope)
            if not args.no_augment:
                env_data = augment_dataset(env_data, rng)
            env, agents, task_time = env_data
            steps = min(args.imitation_steps,
                        args.max_steps or int(args.time_limit / args.single_step))
            for step in range(1, steps + 1):
                center = select_optimal_region(env, agents, center_scope,
                                               step, args.single_step)
                graph = call_silently(
                    not args.verbose, build_graph, env, agents, task_time, *center,
                    center_scope, step, args.single_step, args.time_limit,
                )
                graph = creatWeight(graph, env, agents, task_time,
                                    step, args.single_step)
                if not graph:
                    continue
                labels_set, _ = call_silently(not args.verbose, solve_graph, graph)
                train_encoders = len(graph) <= args.max_imitation_grad_nodes
                encoder_frozen_samples += int(not train_encoders)
                state = pruning_env.extract_state(
                    graph, agents, env, task_time, *center, center_scope,
                    step, args.single_step, args.time_limit, grad=train_encoders,
                )
                z_llm, z_gnn, node_emb, node_mask, node_to_idx = state
                h_fused, _ = agent.fusion(z_llm, z_gnn)
                probs = agent.actor(h_fused, node_emb, node_mask)
                labels = torch.zeros_like(probs)
                for node in labels_set:
                    labels[0, node_to_idx[node]] = 1.0
                positives = labels.sum().clamp(min=1)
                negatives = (node_mask.sum() - labels.sum()).clamp(min=1)
                weights = torch.where(labels > 0, negatives / positives,
                                      torch.ones_like(labels)) * node_mask
                loss = (F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6),
                                               labels, reduction='none') * weights).sum()
                loss = loss / weights.sum().clamp(min=1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()
                losses.append(float(loss.item()))
                epoch_losses.append(float(loss.item()))
                samples += 1

                # Advance along the unpruned expert trajectory.
                selected = sorted(labels_set, key=lambda n: graph.nodes[n]['weight'], reverse=True)
                env, agents, task_time, *_ = call_silently(
                    not args.verbose, updateEnv, env, agents, task_time,
                    selected, step, args.single_step,
                )
        record = {'epoch': epoch + 1, 'samples': samples,
                  'encoder_frozen_samples': encoder_frozen_samples,
                  'loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                  'elapsed': time.perf_counter() - epoch_started}
        history.append(record)
        emit(args, f'  epoch {epoch + 1}/{args.imitation_epochs} | '
                   f'样本={samples} | 大图冻结编码器={encoder_frozen_samples} | '
                   f'平均loss={record["loss"]:.4f} | '
                   f'耗时={record["elapsed"]:.1f}s')

    for module in (llm, gnn):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return {'epochs': history, 'mean_loss': float(np.mean(losses)) if losses else 0.0}


def store_transitions(agent, transitions, terminal_bonus=0.0):
    agent.clear_buffer()
    for index, item in enumerate(transitions):
        reward = item['reward'] + (terminal_bonus if index == len(transitions) - 1 else 0.0)
        agent.store_transition(item['z_llm'], item['z_gnn'], item['node_emb'],
                               item['node_mask'], item['keep_mask'],
                               item['old_log_probs'], reward, item['done'])


def train(paths, components, args):
    llm, gnn, agent, pruning_env = components
    train_started = time.perf_counter()
    emit(args, '\n[阶段 1/3] 模仿预训练')
    if args.imitation_epochs == 0:
        emit(args, '  已跳过（imitation-epochs=0）')
    imitation = imitation_pretrain(paths, components, args)
    call_silently(not args.verbose, agent.save_model, args.model_prefix,
                  llm, gnn, {'phase': 'imitation'})
    emit(args, '\n[阶段 2/3] PPO微调')
    rng = np.random.default_rng(args.seed + 1)
    baselines = {}
    best_score, stale = -math.inf, 0
    training_order = list(paths)
    random.shuffle(training_order)
    episode_history = []
    validation_history = []

    for episode in range(1, args.episodes + 1):
        episode_started = time.perf_counter()
        if (episode - 1) % len(training_order) == 0 and episode > 1:
            random.shuffle(training_order)
        path = training_order[(episode - 1) % len(training_order)]
        original = load_dataset(path)
        center_scope = resolve_center_scope(path, original[0], args.center_scope)
        if path not in baselines:
            baselines[path], _ = run_episode(
                original, components, args, center_scope=center_scope,
                pruning=False, quiet=True,
            )
        env_data = original if args.no_augment else augment_dataset(original, rng)
        metrics, transitions = run_episode(
            env_data, components, args, center_scope=center_scope,
            pruning=True, collect=True,
            compare_full=True, deterministic=False, quiet=True,
        )
        completion_ratio = relative_quality(metrics['completed'], baselines[path]['completed'])
        terminal_bonus = (completion_ratio
                          - 5.0 * max(0.0, args.quality_threshold - completion_ratio))
        store_transitions(agent, transitions, terminal_bonus)
        loss, gate = agent.update()
        episode_record = {
            'episode': episode, 'dataset': path.name, 'loss': loss,
            'weight_ratio': metrics['weight_ratio'],
            'completion_ratio': completion_ratio, 'keep_ratio': metrics['keep_ratio'],
            'gate': gate, 'elapsed': time.perf_counter() - episode_started,
        }
        episode_history.append(episode_record)
        emit(args, f'  episode {episode:03d}/{args.episodes:03d} | {path.name:<22} '
                   f'loss={loss:.4f} | MWIS={metrics["weight_ratio"]:.1%} | '
                   f'完成={completion_ratio:.1%} | 保留={metrics["keep_ratio"]:.1%} | '
                   f'gate={gate:.3f} | {episode_record["elapsed"]:.1f}s')
        cycle_complete = (episode % len(training_order) == 0 or
                          (args.episodes < len(training_order) and episode == args.episodes))
        if cycle_complete:
            emit(args, '\n[阶段 3/3] 全数据集验证')
            validation = validate_training(paths, baselines, components, args)
            cycle_score = validation['score']
            if cycle_score > best_score:
                best_score, stale = cycle_score, 0
                call_silently(not args.verbose, agent.save_model,
                              args.model_prefix, llm, gnn,
                              {'episode': episode, **validation})
            else:
                stale += len(training_order)
            validation_history.append({'episode': episode, **validation})
            if not args.quiet:
                print_evaluation_table(validation['rows'])
            emit(args, f'验证摘要: 最低MWIS={validation["min_weight_ratio"]:.1%} | '
                       f'最低完成={validation["min_completion_ratio"]:.1%} | '
                       f'平均MWIS={validation["mean_weight_ratio"]:.1%} | '
                       f'平均完成={validation["mean_completion_ratio"]:.1%} | '
                       f'平均保留={validation["mean_keep_ratio"]:.1%} | '
                       f'总体验收={"通过" if validation["overall_passed"] else "未通过"}')
            if stale >= args.patience:
                emit(args, f'早停: 连续{stale}个episode无改进')
                break
            emit(args, '\n[阶段 2/3] 继续PPO微调')

    summary = {
        'episodes_completed': len(episode_history), 'best_score': best_score,
        'checkpoint': args.model_prefix, 'elapsed': time.perf_counter() - train_started,
    }
    emit(args, f'\n训练完成: episodes={summary["episodes_completed"]} | '
               f'最佳分数={best_score:.4f} | 总耗时={summary["elapsed"]:.1f}s | '
               f'checkpoint={args.model_prefix}', final=True)
    return {'imitation': imitation, 'episodes': episode_history,
            'validations': validation_history, 'summary': summary}


def validate_training(paths, baselines, components, args):
    rows = []
    for index, path in enumerate(paths, 1):
        emit(args, f'  验证 {index:02d}/{len(paths):02d}: {path.name}')
        data = load_dataset(path)
        center_scope = resolve_center_scope(path, data[0], args.center_scope)
        if path not in baselines:
            baselines[path], _ = run_episode(
                data, components, args, center_scope=center_scope,
                pruning=False, quiet=True,
            )
        pruned, _ = run_episode(data, components, args,
                                center_scope=center_scope, pruning=True,
                                compare_full=True, deterministic=True, quiet=True)
        completion = relative_quality(pruned['completed'], baselines[path]['completed'])
        passed = (pruned['weight_ratio'] >= args.min_dataset_quality and
                  completion >= args.min_dataset_quality)
        rows.append({'dataset': path.name, 'weight_ratio': pruned['weight_ratio'],
                     'completion_ratio': completion, 'keep_ratio': pruned['keep_ratio'],
                     'EE': pruned['EE'], 'ATRT': pruned['ATRT'],
                     'fallbacks': pruned['fallback_count'], 'passed': passed})
    summary = summarize_evaluation(rows, args)
    margins = (
        summary['min_weight_ratio'] - args.min_dataset_quality,
        summary['min_completion_ratio'] - args.min_dataset_quality,
        summary['mean_weight_ratio'] - args.mean_quality_threshold,
        summary['mean_completion_ratio'] - args.mean_quality_threshold,
        args.keep_ratio - summary['mean_keep_ratio'],
    )
    return {'score': min(margins), **summary, 'rows': rows}


def summarize_evaluation(rows, args):
    weights = [row['weight_ratio'] for row in rows]
    completions = [row['completion_ratio'] for row in rows]
    keeps = [row['keep_ratio'] for row in rows]
    minimums_passed = all(row['passed'] for row in rows)
    mean_weight = float(np.mean(weights)) if weights else 0.0
    mean_completion = float(np.mean(completions)) if completions else 0.0
    means_passed = (mean_weight >= args.mean_quality_threshold and
                    mean_completion >= args.mean_quality_threshold)
    mean_keep = float(np.mean(keeps)) if keeps else 0.0
    quality_passed = minimums_passed and means_passed
    keep_passed = mean_keep <= args.keep_ratio
    return {
        'datasets': len(rows),
        'passed': sum(row['passed'] for row in rows),
        'min_weight_ratio': min(weights, default=0.0),
        'min_completion_ratio': min(completions, default=0.0),
        'mean_weight_ratio': mean_weight,
        'mean_completion_ratio': mean_completion,
        'mean_keep_ratio': mean_keep,
        'all_dataset_minimums_passed': minimums_passed,
        'mean_quality_passed': means_passed,
        'all_quality_passed': quality_passed,
        'keep_ratio_passed': keep_passed,
        'overall_passed': quality_passed and keep_passed,
    }


def print_evaluation_table(rows):
    table_rows = [
        [row['dataset'], f'{row["weight_ratio"]:.1%}',
         f'{row["completion_ratio"]:.1%}', f'{row["keep_ratio"]:.1%}',
         f'{row["EE"]:.4f}', f'{row["ATRT"]:.2f}', row['fallbacks'],
         '通过' if row['passed'] else '未通过']
        for row in rows
    ]
    print(format_table(
        ['数据集', 'MWIS比例', '完成比例', '保留率', 'EE', 'ATRT', '回退', '状态'],
        table_rows, ['<', '>', '>', '>', '>', '>', '>', '<'],
    ))


def evaluate(paths, components, args):
    rows = []
    started = time.perf_counter()
    emit(args, '\n[评估] 逐数据集比较剪枝与未剪枝基线')
    for index, path in enumerate(paths, 1):
        emit(args, f'  评估 {index:02d}/{len(paths):02d}: {path.name}')
        data = load_dataset(path)
        center_scope = resolve_center_scope(path, data[0], args.center_scope)
        baseline, _ = run_episode(data, components, args,
                                  center_scope=center_scope,
                                  pruning=False, quiet=True)
        pruned, _ = run_episode(data, components, args,
                                center_scope=center_scope, pruning=True,
                                compare_full=True, deterministic=True, quiet=True)
        completion_ratio = relative_quality(pruned['completed'], baseline['completed'])
        passed = (pruned['weight_ratio'] >= args.min_dataset_quality and
                  completion_ratio >= args.min_dataset_quality)
        row = {'dataset': path.name, 'weight_ratio': pruned['weight_ratio'],
               'completion_ratio': completion_ratio, 'keep_ratio': pruned['keep_ratio'],
               'EE': pruned['EE'], 'ATRT': pruned['ATRT'],
               'fallbacks': pruned['fallback_count'], 'passed': passed}
        rows.append(row)
    if not args.quiet:
        print()
        print_evaluation_table(rows)
    summary = summarize_evaluation(rows, args)
    summary['elapsed'] = time.perf_counter() - started
    emit(args, f'\n评估完成: 通过={summary["passed"]}/{summary["datasets"]} | '
               f'平均MWIS={summary["mean_weight_ratio"]:.1%} | '
               f'平均完成={summary["mean_completion_ratio"]:.1%} | '
               f'平均保留={summary["mean_keep_ratio"]:.1%} | '
               f'总体验收={"通过" if summary["overall_passed"] else "未通过"} | '
               f'耗时={summary["elapsed"]:.1f}s', final=True)
    return {'rows': rows, 'summary': summary}


def main(argv=None):
    args = parse_args(argv)
    thresholds = {
        '--keep-ratio': args.keep_ratio,
        '--quality-threshold': args.quality_threshold,
        '--min-dataset-quality': args.min_dataset_quality,
        '--mean-quality-threshold': args.mean_quality_threshold,
    }
    for name, value in thresholds.items():
        if not 0 < value <= 1:
            raise ValueError(f'{name} must be in (0, 1]')
    if args.max_imitation_grad_nodes < 1:
        raise ValueError('--max-imitation-grad-nodes must be positive')
    if not (args.min_dataset_quality <= args.mean_quality_threshold <=
            args.quality_threshold):
        raise ValueError('expected min-dataset-quality <= mean-quality-threshold '
                         '<= quality-threshold')
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths()
    print_run_header(args, device, paths)
    components = call_silently(not args.verbose, create_components, device)
    llm, gnn, agent, pruning_env = components
    config = run_config(args, device)
    config['dataset_count'] = 1 if args.mode == 'infer' else len(paths)

    if args.mode == 'train':
        result = train(paths, components, args)
        write_output(args, {'config': config, **result})
        return 0

    call_silently(not args.verbose, agent.load_model,
                  args.model_prefix, llm, gnn)
    llm.eval(); gnn.eval(); agent.fusion.eval(); agent.actor.eval(); agent.critic.eval()
    if args.mode == 'eval':
        result = evaluate(paths, components, args)
        write_output(args, {'config': config, **result})
        summary = result['summary']
        return 0 if summary['overall_passed'] else 2

    path = resolve_data(args.data)
    data = load_dataset(path)
    center_scope = resolve_center_scope(path, data[0], args.center_scope)
    started = time.perf_counter()
    emit(args, f'\n[推理] 数据集={path.name} | MDC窗口={center_scope}')
    metrics, _ = run_episode(data, components, args,
                             center_scope=center_scope, pruning=True,
                             compare_full=True,
                             deterministic=True)
    elapsed = time.perf_counter() - started
    summary = {
        'dataset': path.name, 'completed': metrics['completed'],
        'total_tasks': metrics['total_tasks'], 'EE': metrics['EE'],
        'ATRT': metrics['ATRT'], 'charge_count': metrics['charge_count'],
        'weight_ratio': metrics['weight_ratio'], 'keep_ratio': metrics['keep_ratio'],
        'fallback_count': metrics['fallback_count'], 'elapsed': elapsed,
    }
    emit(args, f'\n推理完成: 任务={summary["completed"]}/{summary["total_tasks"]} | '
               f'MWIS={summary["weight_ratio"]:.1%} | 保留={summary["keep_ratio"]:.1%} | '
               f'EE={summary["EE"]:.4f} | ATRT={summary["ATRT"]:.2f} | '
               f'充电={summary["charge_count"]} | 回退={summary["fallback_count"]} | '
               f'耗时={elapsed:.1f}s', final=True)
    write_output(args, {'config': config, 'steps': metrics.pop('steps'),
                        'metrics': metrics, 'summary': summary})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
