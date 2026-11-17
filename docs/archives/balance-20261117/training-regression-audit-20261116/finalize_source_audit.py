"""Compile read-only source findings; no ML/simulator imports or rollouts."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
repo=root.parents[2]
experiment=repo/'tasks/results/aligned-limit-ablation-20261116'
archive=experiment/'comparison/actual-experiment-source.tar.gz'
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
verification=json.loads((experiment/'comparison/verification.json').read_text())
assert archive_sha==verification['source_archive_sha256']
with tarfile.open(archive) as stream:
    manifest=json.loads(stream.extractfile('inputs/manifest-limit1.json').read())
    entries={m.name:hashlib.sha256(stream.extractfile(m).read()).hexdigest() for m in stream.getmembers() if m.isfile()}
    alignment=json.loads(stream.extractfile('inputs/alignment-validation.json').read())

weights={
    'reward/tracking_lin_vel':2.,'reward/tracking_ang_vel':2.,
    'reward/tracking_wheels_sinposition':10.,'reward/tracking_sinyaw':10.,
    'reward/pitch_cost':0.,'reward/roll_cost':4.,'reward/tracking_height_jnt':2.,
    'reward/energy_cost':-1.,'reward/action_rate':-.5,
    'reward/pitch_velocity_penalty':-.5,'reward/termination_penalty':0.,
}
lib='$HOME/projects/tenny-reference/.venv-gpu/lib/python3.12/site-packages'
observed_library_hashes={
    'brax/training/agents/ppo/losses.py':'b6dfe0aefeb1e3d11252fc40c6ec4512757fc7fccc0b3e37ec9ccd349351007c',
    'brax/envs/wrappers/training.py':'8826de9e70f14d97afe9e7c87496f17c80744f356021f7127839c6248c0d7c55',
    'mujoco_playground/_src/wrapper.py':'23c072737b9895c52d61372d67f5359c77613f71a4f363f80681ac48ba79aa3d',
    'brax/training/distribution.py':'3f9375b179f55ebb76fb337db610fafbf60b53b3d329a2fbbfae365836b1d220',
    'brax/training/acme/running_statistics.py':'9591566ebcd151d35dd1a9167de3b1b00df92100892b5889bc9514252fb8c0c4',
}
gamma=manifest['ppo']['discounting'];lam=manifest['ppo']['gae_lambda'];dt=manifest['env_config']['ctrl_dt']
result={
    'schema_version':1,'kind':'training_regression_source_audit',
    'scope':{
        'experiment':'aligned-limit-ablation-20261116',
        'applies_to':'Six short PPO runs initialized from the verified BC bootstrap, both capacity arms newly trained.',
        'does_not_identify_or_audit_user_mature_policy':True,
        'mature_policy_critic_std_training_history_unknown':True,
        'no_new_training_or_rollouts':True,
        'no_running_or_production_source_changed':True,
        'source_review_only_not_causal_identification':True,
    },
    'provenance':{
        'archive_path':str(archive),'archive_sha256':archive_sha,
        'archive_digest_matches_completed_verification':True,
        'archive_members_sha256':entries,
        'protected_runtime_files_read_and_hash_matched':{
            p:d for p,d in manifest['protected_inputs']['files'].items()
            if p.endswith(('/legwheel/env.py','/legwheel/rewards.py','/hybrid/env.py',
                           '/force_teacher/env.py','/brax/training/agents/ppo/train.py','/verified.json'))
        },
        'installed_dependencies_read_now_not_archived_or_formally_hash_frozen':{
            lib+'/'+p:d for p,d in observed_library_hashes.items()},
        'bootstrap_source_snapshot':{
            'name':'source-1-pretrain.py',
            'sha256':'daf9d71ff7311e9ab2b38eabe395ffa32d4eeaa4e526683e3cc2c16e6f69d0de',
            'hash_matches_protected_verified_json_files_entry':True,
        },
    },
    'reward':{
        'native_source_member':'native/src/cpp/legwheel_task.cpp',
        'native_reward_source_lines':[593,637],
        'native_metrics_are_preweight':True,
        'metric_weights':weights,
        'control_reward_factor':dt/manifest['env_config']['reward_config']['reference_dt'],
        'ppo_reward_scaling':manifest['ppo']['reward_scaling'],
        'native_return_is_not_ppo_scaled':True,
        'metric_weighted_reconstruction_formula':'return = ctrl_dt/reference_dt * sum(weight[name] * native_metric_sum[name])',
        'numerical_40_case_reconstruction_not_repeated_by_this_subtask':True,
        'pitch_scale':0.,'sinpitch_zero_does_not_enable_pitch_reward':True,
        'actual_world_xy_position_in_reward':False,
        'position_reward_source':'Uncompensated mean relative wheel joint angle, s=(q_left+q_right)/2; not base world XY.',
        'pose_and_xy_ablation_weights':manifest['arms']['baseline'],
        'termination_penalty_enable':False,
        'configured_but_disabled_terminal_penalty_value':-10.,
        'maximum_positive_reward_per_control_step_without_costs':30.,
        'effort':'Mean over 4 physics substeps and 4 actuators of (actual_torque/reference_scale)^2.',
        'action_rate':'Mean over 4 substeps and actuators of |u_t-u_t-1|*4 + |u_t+u_t-2-2*u_t-1|*16, u=tau/reference_scale.',
        'reference_scales_nm':[1.,1.,.0295,.0295],
        'indirect_balance_constraints':['roll tracking','mean hip-height tracking','local forward/yaw velocity tracking','pitch angular velocity cost','loss of future rewards on termination'],
        'source_proves_objective_mismatch_but_not_that_it_caused_observed_regression':True,
        'raw_sensor_phase':'Native xmat/sensordata remain at MuJoCo mj_step derived stage; post-integration qpos/qvel are not identical-time sensor samples.',
        'npz_cannot_reconstruct_full_substep_effort_or_rate_from_final_ctrl_alone':True,
    },
    'ppo':{
        'explicit_options':manifest['ppo'],
        'default_options_verified_in_protected_train_source':{
            'normalize_advantage':True,'vf_loss_coefficient':.5,
            'clipping_epsilon_value':None,'bootstrap_on_timeout':False,
            'use_distributional_critic':False,'learning_rate_schedule':None,'restore_value_fn':True},
        'gamma_half_life_seconds':math.log(.5)/math.log(gamma)*dt,
        'gae_trace_gamma_lambda_half_life_seconds':math.log(.5)/math.log(gamma*lam)*dt,
        'unroll_seconds':manifest['ppo']['unroll_length']*dt,
        'gamma_power_16':gamma**16,'gamma_lambda_power_16':(gamma*lam)**16,
        'loss_terms':'PPO ratio clipped surrogate + .5*vf_loss_coefficient*MSE(GAE_target,V) - entropy_cost*entropy.',
        'ratio_clip_interval':[.7,1.3],
        'ratio_clip_is_not_a_hard_action_or_KL_bound':True,
        'single_update_clip_scope':'With 1 minibatch, 1 update and fixed normalization, the sole gradient evaluation is at the behavior actor (rho approximately 1); ratio clipping does not bound the resulting post-update actor drift. This is a control-flow inference, not a measured clipping statistic.',
        'optimizer':'Fresh Adam, lr=3e-5, global gradient norm clip=1 on policy/value parameter tree.',
        'no_bc_or_frozen_anchor_kl_regularization_in_loss':True,
        'frozen_anchor_is_a_mean_parameterization_not_a_behavior_retention_penalty':True,
        'full_30_second_monte_carlo_target_not_used':True,
        'boundary_bootstrap':'V of last next_observation is used at 16-step rollout boundary; true termination cuts bootstrapping.',
        'truncation':'EpisodeWrapper distinguishes timeout from environment failure; GAE masks timeout transition and trace. Explicit timeout reward-bootstrap correction is disabled by default.',
        'native_first_episode_termination_wins_over_timeout':True,
        'likelihood':'Both rollout and loss use replaced network.apply; target log_prob uses saved pre-cap raw_action. No additional parameterization Jacobian needed.',
        'checkpoint_selection':'Trainer saves final actor/critic; common physical standing metric does not gate or roll back PPO updates.',
    },
    'initialization':{
        'checkpoint':manifest['initial_checkpoint'],
        'checkpoint_sha256':manifest['initial_checkpoint_sha256'],
        'complete_array_digest':manifest['initial_array_digest'],
        'bootstrap_critic_receipt':'Random untrained value network, fitted observation normalization only',
        'bc_objective_receipt':'Mean squared force/Fscale and wheel torque/0.0295 errors, equal weight per normalized actuator; never mixes raw N with raw Nm',
        'bc_candidate_fit_updates':2048,'bc_candidate_samples':13404,'bc_candidate_episodes':72,
        'candidate_fit_std_before_bootstrap_reset':.15,
        'actual_initial_std_final_protected_receipt':.01,
        'std_source':'verified.json.provenance.exploration_reset.pre_tanh_std and identity.exploration_pre_tanh_std, both .01; not the older bc.initial_exploration_pre_tanh_std=.15.',
        'checkpoint_std_was_not_newly_evaluated_or_loaded_by_this_source_subtask':True,
        'std_head_semantics':'Zero scale-head kernels, inverse-softplus bias to .01 pre-tanh std; trainable again in PPO.',
        'normalization':'EMA fixed mean/std/variance with until_count=0; count may grow but does not enter normalize().',
        'completed_run_normalizer_drift_max':0.,
        'anchor':'Frozen BC actor closure, outside optimizer parameter tree; raw scale logits and critic are unchanged by common mean transform.',
        'deterministic_initial_physical_alignment':alignment['deterministic_pair_max_abs_errors'],
        'random_scope':alignment.get('sampling_scope'),
        'random_pre_cap_low_capacity_fraction':alignment.get('initial_random_force_request_over_low_capacity_fraction'),
        'not_permanent_half_amplitude':'Mutable mean residual can use full common action range; supplied probe reached ~.9879 action and ~1.682 Nm in high-capacity arm.',
    },
    'training_reset':{
        'protected_source':'$HOME/projects/tenny-torque200/src/legwheel/env.py:178-201',
        'default_random_flag':False,
        'ppo_reset_call':'env.reset(key_envs), no random=True argument',
        'nominal_qpos0_and_zero_velocity':True,
        'random_orientation_code_exists_but_is_not_selected':True,
        'command_probability':0.,'observation_noise_level':0.,'external_push_enabled':False,
        'domain_randomization_fn':None,
        'autoreset':'Cached first data/obs restored, and HybridEnv resets relevant histories and references on fresh steps.',
        'evaluation_nominal_matches_nominal_training_reset':True,
        'fixed_pitch_perturbations_are_outside_nominal_training_reset_distribution':True,
        'not_evidence_of_large_random_reset_train_eval_mismatch':True,
    },
    'causal_status':{
        'reward_task_mismatch':'Established structurally: direct pitch weight zero and true XY missing.',
        'ppo_update_risk':'Established untrained BC critic and short GAE/unroll bootstrap; bad advantage estimates or excessive actor drift remain hypotheses.',
        'exclusive_cause_not_identified':'The two can coexist. Need common-window native objective and value/advantage diagnostics, not survival alone.',
        'mature_policy':'No conclusion about its critic initialization, std, or fine-tuning stability; identity must be established separately.',
    },
    'minimal_discriminating_diagnostics_not_executed':[
        'Identify the user mature checkpoint first; preserve its actor, critic, normalization, action/head and original training configuration before any new fine-tune.',
        'On shared pre-failure windows compare physical standing and actual weighted reward terms; higher reward with lower standing supports objective tradeoff, lower values of both supports update failure too.',
        'For this BC-start experiment only, compare initial critic predictions against frozen-policy discounted returns/TD errors on the same states; measure advantage sign/order before attributing to critic.',
        'If explicitly authorized later, freeze actor and fit only critic or log one actor update with KL/action-drift/std, using a bounded matched control; do not silently add a new PPO run now.',
    ],
}
(root/'source-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(archive_sha256=archive_sha,archive_members=len(entries),weights=weights,
    gamma_half_life=result['ppo']['gamma_half_life_seconds'],gae_half_life=result['ppo']['gae_trace_gamma_lambda_half_life_seconds'],
    gamma_power_16=result['ppo']['gamma_power_16'],gamma_lambda_power_16=result['ppo']['gamma_lambda_power_16'])))
