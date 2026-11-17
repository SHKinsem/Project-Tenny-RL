"""Read-only reconstruction of saved training/evaluation samples; no ML imports."""
import hashlib
import ast
import json
import math
from pathlib import Path
import statistics
import tarfile
import time


STEPS = [0, 524288, 1048576, 1572864, 2097152]
METRICS = ('training/entropy_loss', 'training/kl_mean', 'training/learning_rate',
    'training/policy_dist_max_loc', 'training/policy_dist_max_std',
    'training/policy_dist_mean_loc', 'training/policy_dist_mean_std',
    'training/policy_dist_min_loc', 'training/policy_dist_min_std',
    'training/policy_loss', 'training/total_loss', 'training/v_loss')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    return math.fsum(values)/len(values)


def optimizer_evidence(manifest, runs):
    """Inspect source text only; importing Brax or constructing an optimizer is unnecessary."""
    source_path, expected = next((path, digest) for path, digest in manifest['protected_inputs']['files'].items()
        if path.endswith('/brax/training/agents/ppo/train.py'))
    configured_schedules = [run['configured_ppo'].get('learning_rate_schedule') for run in runs]
    assert all(value is None for value in configured_schedules)
    logged_rates = sorted({row['metrics']['training/learning_rate'] for run in runs for row in run['training_metric_samples']})
    assert len(logged_rates) == 1
    evidence = dict(configured_learning_rate=3e-5, logged_sample_count=24,
        logged_unique_values=logged_rates, configured_learning_rate_schedules=configured_schedules,
        source_path=source_path, expected_source_sha256=expected,
        source_available_to_this_process=Path(source_path).is_file())
    if evidence['source_available_to_this_process']:
        assert sha(source_path) == expected
        source = Path(source_path).read_text()
        parsed = ast.parse(source)
        train = next(node for node in parsed.body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)) and node.name == 'train')
        positional = dict(zip([arg.arg for arg in train.args.args[-len(train.args.defaults):]], train.args.defaults))
        defaults = dict(positional, **{arg.arg: value for arg,value in zip(train.args.kwonlyargs,train.args.kw_defaults)})
        schedule = defaults['learning_rate_schedule']
        assert isinstance(schedule, ast.Constant) and schedule.value is None
        assert not any(isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id=='learning_rate' for node in ast.walk(train))
        stripped = {line.strip() for line in source.splitlines()}
        assert 'base_optimizer = optax.adam(learning_rate=learning_rate)' in stripped
        assert 'lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE' in stripped
        assert 'lr = jnp.array(learning_rate)' in stripped
        evidence.update(source_sha256_verified=True, default_schedule=None, selected_schedule='NONE',
            optimizer='Adam with fixed numeric learning_rate; global-norm clipping configured at 1.',
            adaptive_kl_branch_selected=False,
            snippets=[dict(line=i,text=line.strip()) for i,line in enumerate(source.splitlines(),1)
                if any(word in line for word in ('learning_rate: float','base_optimizer =','lr_schedule =','lr_is_adaptive_kl =','lr = jnp.array(learning_rate)','metrics[\'learning_rate\']'))])
    return evidence


def analyze_run(path, comparison):
    condition, _, raw_seed = path.name.partition('-seed-')
    seed = int(raw_seed)
    progress = read_jsonl(path/'progress.jsonl')
    training = read_jsonl(path/'training-metrics.jsonl')
    config, status = read_json(path/'effective-config.json'), read_json(path/'status.json')
    assert [row['steps'] for row in progress] == STEPS, path
    assert [row['steps'] for row in training] == STEPS[1:], path
    assert status['phase'] == 'completed' and status['steps'] == STEPS[-1]
    assert config['seed'] == seed and config['ppo']['seed'] == seed
    assert all(len(row['common_episode_seconds']) == 3 for row in progress)
    assert all(row['normalizer_drift'] == 0 for row in progress)
    assert all(0 < value <= 6 for row in progress for value in row['common_episode_seconds'])
    assert all(finite(row['common_base_reward_mean']) and finite(row['elapsed_seconds']) for row in progress)
    assert all(finite(value) for row in training for value in row['metrics'].values())
    assert all(set(METRICS).issubset(row['metrics']) for row in training)
    assert status['normalizer_drift'] == 0 and status['actor_changed'] and status['checkpoint_reload_exact']
    lookup = {row['steps']: row['metrics'] for row in training}
    baseline = progress[0]['common_base_reward_mean']
    evaluations = []
    for row in progress:
        value = row['common_base_reward_mean']
        evaluations.append(dict(steps=row['steps'], elapsed_seconds=row['elapsed_seconds'],
            deterministic_6s_return_mean=value, return_delta_from_zero=value-baseline,
            return_percent_from_zero=100*(value/baseline-1),
            first_episode_seconds=list(row['common_episode_seconds']),
            mean_first_episode_seconds=mean(row['common_episode_seconds']),
            capped_at_6s_count=sum(math.isclose(v, 6., rel_tol=0, abs_tol=1e-9) for v in row['common_episode_seconds']),
            actor_and_normalizer_sha256=row['actor_and_normalizer_sha256'],
            training_metrics=lookup.get(row['steps']),
            standing_metrics=None))
    changes = []
    for left, right in zip(evaluations, evaluations[1:]):
        changes.append(dict(from_steps=left['steps'], to_steps=right['steps'],
            return_delta=right['deterministic_6s_return_mean']-left['deterministic_6s_return_mean'],
            mean_short_survival_delta=right['mean_first_episode_seconds']-left['mean_first_episode_seconds']))
    loss_series = {name: [dict(steps=row['steps'], value=row['metrics'][name]) for row in training] for name in METRICS}
    metric_ranges = {name: dict(first=values[0]['value'], last=values[-1]['value'],
        minimum=min(row['value'] for row in values), maximum=max(row['value'] for row in values))
        for name, values in loss_series.items()}
    current = next(row for row in comparison['conditions'][condition]['seed_results'] if row['seed'] == seed)
    zero = comparison['initial'][condition]['primary']
    final_band = current['primary']['mean_in_band_seconds']
    declines = [row['steps'] for row in evaluations[1:] if row['deterministic_6s_return_mean'] < baseline]
    recoveries = [row['steps'] for row in evaluations[1:] if row['deterministic_6s_return_mean'] >= baseline]
    return dict(policy_id=path.name, condition=condition, seed=seed, evaluation_samples=evaluations,
        training_metric_samples=training, loss_series=loss_series, metric_ranges=metric_ranges,
        consecutive_evaluation_changes=changes,
        first_observed_return_below_zero_steps=declines[0] if declines else None,
        post_zero_at_or_above_initial_return_steps=recoveries,
        reward_rebound_steps=[row['to_steps'] for row in changes[1:] if row['return_delta'] > 0],
        best_observed_post_zero_return=max(row['deterministic_6s_return_mean'] for row in evaluations[1:]),
        final_return_percent_from_zero=evaluations[-1]['return_percent_from_zero'],
        final_long_evaluation=dict(primary_cumulative_in_band_seconds=final_band,
            primary_longest_in_band_seconds=current['primary']['mean_longest_in_band_seconds'],
            primary_survival_seconds=current['primary']['mean_survival_seconds'],
            initial_primary_cumulative_in_band_seconds=zero['mean_in_band_seconds'],
            delta_cumulative_from_zero=final_band-zero['mean_in_band_seconds']),
        snapshot_evidence=dict(windows_files=sorted(item.name for item in path.iterdir()),
            save_checkpoint_path=config['ppo']['save_checkpoint_path'],
            intermediate_weight_arrays_in_windows_archive=False,
            sampled_actor_hash_count=len({row['actor_and_normalizer_sha256'] for row in progress}),
            final_actor_changed=status['actor_changed'],
            additional_wsl_manual_artifacts_checked=False),
        configured_ppo=config['ppo'],
        normalization_drift_max=max(row['normalizer_drift'] for row in progress))


def analyze(root):
    started = time.monotonic()
    source = root/'tasks/results/aligned-limit-ablation-20261116'
    comparison = read_json(source/'comparison/paired-summary.json')
    inputs = {str(source/'comparison/paired-summary.json'): sha(source/'comparison/paired-summary.json')}
    archive = source/'comparison/actual-experiment-source.tar.gz'
    inputs[str(archive)] = sha(archive)
    with tarfile.open(archive) as stream:
        member = stream.getmember('experiments/reward_ablation/train.py')
        data = stream.extractfile(member).read()
    runtime = read_json(source/'inputs/runtime-freeze.json')
    expected = next(value for path, value in runtime.items() if path.endswith('/experiments/reward_ablation/train.py'))
    assert hashlib.sha256(data).hexdigest() == expected
    lines = data.decode('utf-8').splitlines()
    source_snippets = [dict(line=i, text=line.strip()) for i, line in enumerate(lines, 1)
        if any(text in line for text in ('states =', 'Evaluator(current)', 'steps=600',
            'latest-descriptor.json', 'checkpoint =', 'checkpointer.save', 'np.isfinite(value)'))]
    assert any('steps=600' in row['text'] and 'episode_steps=600' in row['text'] for row in source_snippets)
    assert sum('checkpointer.save' in line for line in lines) == 1
    runs = []
    for condition in ('limit1', 'limit2'):
        for seed in (0, 1, 2):
            path = source/'runs'/f'{condition}-seed-{seed}'
            for filename in ('progress.jsonl', 'training-metrics.jsonl', 'effective-config.json', 'status.json'):
                inputs[str(path/filename)] = sha(path/filename)
            runs.append(analyze_run(path, comparison))
    configs = [{key: value for key, value in run['configured_ppo'].items() if key != 'seed'} for run in runs]
    assert all(config == configs[0] for config in configs)
    assert all(run['evaluation_samples'][0]['deterministic_6s_return_mean'] == runs[0]['evaluation_samples'][0]['deterministic_6s_return_mean'] for run in runs)
    all_metrics = [row['metrics'] for run in runs for row in run['training_metric_samples']]
    aggregate_ranges = {name: dict(minimum=min(row[name] for row in all_metrics),
                                  maximum=max(row[name] for row in all_metrics)) for name in METRICS}
    first_value_down = sum(run['metric_ranges']['training/v_loss']['last'] < run['metric_ranges']['training/v_loss']['first'] for run in runs)
    first_to_second_value_down = sum(run['loss_series']['training/v_loss'][1]['value'] < run['loss_series']['training/v_loss'][0]['value'] for run in runs)
    return_first_to_second_up = sum(run['evaluation_samples'][2]['deterministic_6s_return_mean'] > run['evaluation_samples'][1]['deterministic_6s_return_mean'] for run in runs)
    manifest = read_json(source/'inputs/manifest-limit1.json')
    learning_rate_evidence = optimizer_evidence(manifest,runs)
    inputs[str(source/'inputs/manifest-limit1.json')] = sha(source/'inputs/manifest-limit1.json')
    if learning_rate_evidence.get('source_sha256_verified'):
        inputs[learning_rate_evidence['source_path']] = learning_rate_evidence['expected_source_sha256']
    summary = dict(schema_version=1, kind='saved_training_curve_audit', run_count=6,
        scope=dict(initialization='Common bootstrap BC actor/critic/normalizer; six fresh paired short training runs.',
            initial_checkpoint=manifest['initial_checkpoint'],
            mature_balancing_model_fine_tuning_examined=False,
            restriction='Results apply only to these six BC-start experiments, not to the user\'s previously working mature balancing policy or low-learning-rate position fine tuning.'),
        learning_rate_evidence=learning_rate_evidence,
        observation_contract=dict(evaluation_steps=STEPS, training_metric_steps=STEPS[1:],
            evaluation='Deterministic native first-episode return, capped at 6 seconds on the same three primary initial states.',
            training_episode_seconds=30, fixed_normalizer=True,
            no_zero_step_training_metrics=True, no_intermediate_standing_metrics=True),
        runs=runs, aggregate=dict(metric_ranges=aggregate_ranges,
            initial_deterministic_6s_return=runs[0]['evaluation_samples'][0]['deterministic_6s_return_mean'],
            final_return_lower_than_initial_runs=sum(run['final_return_percent_from_zero'] < 0 for run in runs),
            first_524288_return_lower_than_initial_runs=sum(run['first_observed_return_below_zero_steps'] == 524288 for run in runs),
            final_return_percent_range=[min(run['final_return_percent_from_zero'] for run in runs), max(run['final_return_percent_from_zero'] for run in runs)],
            final_training_value_loss_lower_than_first_training_sample_runs=first_value_down,
            value_loss_lower_at_1048576_than_524288_runs=first_to_second_value_down,
            deterministic_return_higher_over_that_same_interval_runs=return_first_to_second_up,
            final_short_evaluation_all_three_capped_runs=[run['policy_id'] for run in runs if run['evaluation_samples'][-1]['capped_at_6s_count'] == 3],
            final_return_up_but_standing_down_runs=[run['policy_id'] for run in runs if run['final_return_percent_from_zero'] > 0 and run['final_long_evaluation']['delta_cumulative_from_zero'] < 0]),
        effective_time_scales=dict(control_dt=.01, unroll_steps=configs[0]['unroll_length'],
            unroll_seconds=.01*configs[0]['unroll_length'], discounting=configs[0]['discounting'],
            reward_discount_half_life_seconds=.01*math.log(.5)/math.log(configs[0]['discounting']),
            gae_lambda=configs[0]['gae_lambda'],
            note='Configured time scales are possible follow-up questions, not a cause established by these sparse curves.'),
        source_evidence=dict(executed_training_archive=str(archive), training_member=member.name,
            executed_training_source_sha256=expected, snippets=source_snippets,
            saves_only_one_checkpoint_after_train_returns=True,
            progress_latest_descriptor_is_overwritten=True,
            metrics_writer_filters_nonfinite_scalars=True),
        missing_evidence=['Stochastic training rollout returns, episode lengths, reward components and action/capacity distributions were not logged here.',
            'Clip fraction, actual gradient norms, advantage/TD-error distributions and explained variance are not in these sampled metrics.',
            'No loss/std sample at step zero and no per-update history between the four segment-end samples.',
            'Intermediate long-evaluation physical standing metrics and recoverable intermediate policy snapshots are not in the Windows archive.',
            'No exhaustive search was made for additional manually created WSL snapshots.'],
        interpretation=['All six final deterministic six-second returns are below the shared zero-step return; these curves do not establish rising reward with falling balance.',
            'A decreasing critic loss or a near-zero policy-loss scalar does not demonstrate policy improvement or absent parameter updates.',
            'Entropy-loss is a loss term, not a direct entropy curve; aggregate latent std statistics are not physical torque-noise magnitudes.',
            'Six-second survival can censor an already drifting policy; the final 30-second standing result is a separate measurement.'],
        input_sha256=inputs, elapsed_seconds=time.monotonic()-started)
    assert all(sha(path) == expected for path, expected in inputs.items())
    summary['inputs_unchanged'] = True
    return summary


def markdown(summary):
    lines=['# 六次训练曲线审计', '',
        '**范围仅限六次从共同bootstrap BC初始化出发的配对短训。没有检查用户此前已经正常平衡、仅有轻微漂移的成熟模型，也没有执行该成熟模型的低学习率位置保持微调；以下结果不能外推为该路线无效。**', '',
        '只读既有记录。共同0步的三初态确定性6秒回报为'+f"{summary['aggregate']['initial_deterministic_6s_return']:.6f}"+
        '；六次最终回报全部更低，因而现有曲线不支持“训练把原回报提高，却牺牲原地站立”的整体解释。', '',
        '## 采样口径', '',
        '- progress每run仅5点：0、524288、1048576、1572864、2097152环境步；每点是3个固定主要初态、6秒上限的确定性原生评价，不是随机训练回合收益。训练回合本身为30秒。',
        '- training-metrics每run仅4个段末点，没有0步loss/std。保存loss、熵损失、KL和分布loc/std统计；未保存训练episode回报/长度、reward分项、clip fraction或梯度范数。未采样不能填零。',
        '- 全部normalizer漂移0；各采样actor哈希改变，最终status确认actor有更新和重载精确。', '',
        '## 学习率与标准化', '',
        '六run的PPO配置learning_rate均为3e-5，未设置learning_rate_schedule。24个实际日志采样只有一个值：'+repr(summary['learning_rate_evidence']['logged_unique_values'][0])+'，是3e-5的float32表示，不是动态衰减。', '',
        ('实际安装Brax源码已按保护SHA核验：train的learning_rate_schedule默认None，选择LRSchedule.NONE，Adam直接接收常数learning_rate；自适应KL分支存在但本轮未启用。源码内没有对learning_rate参数重新赋值。' if summary['learning_rate_evidence'].get('source_sha256_verified') else
         '此进程未访问WSL安装源码；这里只确认配置与采样值。完整优化器源证据需在可读取该路径的环境运行本脚本。'), '',
        'PPO标准化为EMA、normalize_until_count=0、std_eps=0；所有过程采样normalizer_drift=0。它不是随着训练继续拟合观测统计的设置。', '',
        '## 回报与终点站立能力', '',
        '| run | 0→末次回报变化 | 首次观察到低于0步的采样步 | 末次6秒平均存活 | 末次触6秒初态数 | 最终30秒累计合格 | 相对0步累计合格变化 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in summary['runs']:
        final=run['evaluation_samples'][-1];long=run['final_long_evaluation']
        lines.append(f"| {run['policy_id']} | {run['final_return_percent_from_zero']:+.3f}% | {run['first_observed_return_below_zero_steps']} | {final['mean_first_episode_seconds']:.6f}s | {final['capped_at_6s_count']}/3 | {long['primary_cumulative_in_band_seconds']:.6f}s | {long['delta_cumulative_from_zero']:+.6f}s |")
    lines += ['', '最终长评估的主要初态累计合格参考为3.970秒，全部六个run都更低。这里链接最终独立评价，不存在逐中间采样点的站立质量曲线。', '',
        '5/6在首个524288步采样已出现回报下降；limit1-seed-2该点10598.7709略高于0步10568.9423，1048576步降至9595.8420。只能定位“首次观察到下降”的区间，不能定位哪个PPO更新最先损坏了策略。', '',
        '| run | 0 / 524288 / 1048576 / 1572864 / 2097152 的确定性回报 |', '| --- | --- |']
    for run in summary['runs']:
        lines.append('| '+run['policy_id']+' | '+' / '.join(f"{row['deterministic_6s_return_mean']:.3f}" for row in run['evaluation_samples'])+' |')
    lines += ['', '各run均有局部回报反弹；除limit1-seed-2首个更新段外，没有后续采样恢复到共同0步回报。limit1-seed-2和limit2-seed-1末次三初态都撑满6秒，但长评估累计合格仍仅3.617/3.267秒，显示短时存活封顶不能代表原地平衡。', '',
        '## 优化指标', '',
        '| run | 四次value loss | policy loss绝对值峰值 | mean std首→末 | max std末次 |', '| --- | --- | ---: | --- | ---: |']
    for run in summary['runs']:
        value=' / '.join(f"{row['value']:.4f}" for row in run['loss_series']['training/v_loss'])
        std=run['metric_ranges']['training/policy_dist_mean_std']
        maxstd=run['metric_ranges']['training/policy_dist_max_std']['last']
        pl=max(abs(row['value']) for row in run['loss_series']['training/policy_loss'])
        lines.append(f"| {run['policy_id']} | {value} | {pl:.3e} | {std['first']:.8f}→{std['last']:.8f} | {maxstd:.8f} |")
    ranges=summary['aggregate']['metric_ranges']
    lines += ['',
        f"- 524288→1048576段，6/6的value loss下降，但确定性回报仅{summary['aggregate']['deterministic_return_higher_over_that_same_interval_runs']}/6增加；这段critic拟合改善不能当成policy改善。四次value loss均先降后反弹，最终只有{summary['aggregate']['final_training_value_loss_lower_than_first_training_sample_runs']}/6低于首段。",
        f"- 已记录mean std范围{ranges['training/policy_dist_mean_std']['minimum']:.9f}～{ranges['training/policy_dist_mean_std']['maximum']:.9f}；没有在这些段末均值中观察到数量级爆增或坍缩。最小std最低{ranges['training/policy_dist_min_std']['minimum']:.9f}，最大std最高{ranges['training/policy_dist_max_std']['maximum']:.9f}。这些是latent分布的跨状态/动作聚合，不是每条腿的物理力矩扰动，且没有0步样本。",
        f"- entropy_loss范围{ranges['training/entropy_loss']['minimum']:.9f}～{ranges['training/entropy_loss']['maximum']:.9f}，应保留“loss”含义，不直接称为熵或据其升高断言探索增加。",
        f"- KL日志范围{ranges['training/kl_mean']['minimum']:.12g}～{ranges['training/kl_mean']['maximum']:.12g}；只有4个段末聚合，缺少逐更新KL/clip fraction，不能仅凭小数值排除更新问题。",
        '- policy loss接近零与参数未更新不是同一事实；采样actor哈希和完成记录都证明参数发生变化。缺少advantage/梯度分布，不能由其数量级推导梯度是否正确。', '',
        '## Snapshot与缺失证据', '',
        '实际冻结源码tar中的reward_ablation/train.py已与runtime-freeze SHA核对一致：'+summary['source_evidence']['executed_training_source_sha256']+'。过程回调记录5次哈希、写latest-descriptor.json并覆盖；checkpoint只在train返回后保存一次，PPO配置save_checkpoint_path=null。Windows每run仅归档progress/training-metrics/effective-config/status四文件，没有中间权重。没有穷举WSL额外人工产物。', '',
        '因此可做0步与最终策略的奖励分项/回放诊断，但仅靠目前归档无法回放524288/1048576/1572864步策略，也不能将首次下降精确归因于某项reward/某次更新。训练随机rollout实际回报未记录，确定性回报下降也不等于已证明PPO在优化过程中没有提高其采样目标。', '',
        '配置仍为lr3e-5、entropy_cost=.001、max_grad_norm=1、gamma=.9862327045（奖励折扣半衰期.5秒）、lambda=.9025、unroll16（.16秒）。这些短时间尺度可列为后续受控诊断问题，现有曲线不能证明它们就是退化根因。', '',
        f"一次解析与输入核对主体{summary['elapsed_seconds']:.3f}秒，读取的{len(summary['input_sha256'])}份输入SHA均未变；未启动训练、评估或模型加载。详细每步字段与未采样标记见curves-summary.json。", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    here=Path(__file__).resolve().parent
    summary=analyze(here.parents[2])
    (here/'curves-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (here/'curves-notes.md').write_text(markdown(summary),encoding='utf-8')
    print(json.dumps(dict(runs=summary['run_count'],aggregate=summary['aggregate'],elapsed_seconds=summary['elapsed_seconds']),ensure_ascii=False,indent=2))
