from dataclasses import asdict
import unittest

from experiments.reward_ablation.rewards import ARMS
from experiments.reward_ablation.train import final_policy_metadata, validate_reward_manifest


class RewardArtifactContractTests(unittest.TestCase):
    def test_fixed_manifest_matches_actual_executed_coefficients(self):
        manifest={'arms':{key:asdict(value) for key,value in ARMS.items()}}
        validate_reward_manifest(manifest)
        manifest['arms']['position']['position_weight']=9.
        with self.assertRaisesRegex(ValueError,'coefficients'):
            validate_reward_manifest(manifest)

    def test_new_policy_identity_does_not_claim_initializer_export(self):
        original={'source_fingerprint':'physics','model_run_id':'old-run','export_sha256':'old-export',
                  'kind':'run_initializer','initializer_params_sha256':'bc', 'checkpoint_sha256':'old-checkpoint'}
        result=final_policy_metadata(original,'position',2,2097152,'new-checkpoint','same-start',{})
        self.assertEqual(result['kind'],'saved_reward_ablation_policy')
        self.assertEqual(result['checkpoint_sha256'],'new-checkpoint')
        self.assertEqual(result['model_run_id'],'reward-ablation-v1/position-seed-2')
        self.assertNotIn('export_sha256',result)
        self.assertEqual(result['initializer_provenance']['export_sha256'],'old-export')
        self.assertEqual(original['model_run_id'],'old-run')
        self.assertEqual(result['source_fingerprint'],'physics')
