#ifndef RL_POLICY_H
#define RL_POLICY_H

#define ACTION_SIZE 4
// 与 host/policy/ckpt.py DEPLOY_CKPT 必须相同。仿真和真机都跑这一份。
#define DEPLOY_CKPT "ckpts/newobs-norecovery/000403660800"

class RL_Policy {
    private:
        void *NN;
        float nn_output[8];
        float scale;
        unsigned int *layers;
        byte *Actv_Functions;
        float *input_history;
        float *normalized_history;
        int history_length;
        int one_observation_length;
        int history_obs_length;
        
        float *action_bias;
        float *action_scale;
        float *actions_;

        void updateHistory(const float *new_input);
        void normalizeObservation();
        float *NN_Forward();

    public:
        RL_Policy(int history_len, int one_obs_len);
        ~RL_Policy();
        //new_input: [info["command"], noisy_gyro, noisy_sinpitch, noisy_sinroll, noisy_jnts, noisy_velocity, noisy_linvel, info["last_act"]]
        void cleanHistory();
        void run(const float *new_input, int inputDim, float *action, int actionDim);
        String debugInfo();
        String cvtSensors2String(const float *dataArray, const int arraySize, const int deci);
        void getActionScale(float *action_scale) {
            for (int i = 0; i < ACTION_SIZE; i++) {
                action_scale[i] = this->action_scale[i];
            }
        }
};

#endif // RL_POLICY_H