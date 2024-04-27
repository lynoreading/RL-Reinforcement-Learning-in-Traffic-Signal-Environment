
from env.TrafficEnv import TrafficEnvironment
from stable_baselines3 import A2C


num_phases = 4
steps = 50000


out_csv = 'output_TrafficEnv_a2c_steps_{}'.format(steps)
env = TrafficEnvironment(num_phases=num_phases,
                         out_put_name=out_csv,
                         simulation_step=steps,
                         # fixed=True,
                         seed=42)

model = A2C("MlpPolicy", env, verbose=0, learning_rate=0.001)

run = 1
model.learn((steps+1)*run)
model.save("PPO_define/PPOmodel_trafficenv_a2c_run{}".format(run))
