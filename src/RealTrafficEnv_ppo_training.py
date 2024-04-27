from stable_baselines3 import PPO
from environment.RealTrafficEnv import RealTrafficEnvironment


if __name__ == '__main__':
    num_phases = 8
    steps = 86399
    out_csv = 'output_RealTrafficEnv_steps_{}'.format(steps)
    env = RealTrafficEnvironment(num_phases=num_phases,
                                 out_put_name=out_csv,
                                 simulation_step=steps,
                                 seed=42)

    # print(len(env.lanes))

    model = PPO('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_rate=3e-4,  # 5e-4
                batch_size=32,
                gamma=0.8,
                verbose=1)
    # model = PPO.load("PPO_define\PPOmodel.zip", env=env)

    run = 5
    model.learn(86400 * run)
    mode_path = 'PPO_define/PPOmodel_realtrafficenv_ppo_run{}'.format(
        run)
    model.save(mode_path)
