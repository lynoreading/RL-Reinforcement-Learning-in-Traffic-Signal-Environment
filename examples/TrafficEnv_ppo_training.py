from env.TrafficEnv import TrafficEnvironment
from stable_baselines3 import PPO

num_phases = 4
steps = 50000

out_csv = 'output_TrafficEnv_ppo_steps_{}'.format(steps)
env = TrafficEnvironment(num_phases=num_phases,
                         out_put_name=out_csv,
                         simulation_step=steps,
                         # fixed=True,
                         seed=42)

model = PPO('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[64, 64]),
            learning_rate=5e-4,
            batch_size=32,
            gamma=0.8,
            verbose=1)
run = 1
model.learn((steps+1) * run)
model.save("PPO_define/PPOmodel_trafficenv_ppo_run{}".format(run))
