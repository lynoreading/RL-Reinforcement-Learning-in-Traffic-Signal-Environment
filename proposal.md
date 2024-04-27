# RL Project Proposal
### group name: Reward Chasers, member: Yu Li, Yifan Wang

We intend to use Reinforcement Learning to optimize traffic signal control algorithms, reducing wait times for both pedestrians and vehicles at road intersections (Track I). As a specific example, consider a fourway intersection where Pfarrstrasse and Ricklingerstadtweg meet. The implementation of the approach should required as follows:
1. Meeting the fundamental functionality of traffic lights. 
2. Ensuring that trams do not need to stop at red lights (i.e., when a tram approaches, the light on the corresponding direction must be green). 
3. When a pedestrian presses the button, the traffic light should change to green for that direction within a certain timeframe.


We plan to represent all possible traffic light states using a one-hot vector. The actions of the agent could be selecting the next green light direction and determining its duration. The ultimate goal is to reduce wait times for both pedestrians and vehicles. The challenge before implementation of RL might lie in developing a simulator capable of emulating the interaction among vehicles, pedestrians, and trams. There is a simulator called ’traci’ that can, to some extent, simulate vehicles behavior. However, creating suitable simulations for trams and pedestrians requires alternative approaches.
After the design of process and reward function, we aim to train the agent using algorithms like SAC (Soft Actor-Critic), PPO (Proximal Policy Optimization), or SARSA. At the end, we will compare the results of the agent’s performance with actual traffic signal control results.

As for the next step, we expect to gain insights like: Analyzation of the decision- making process of the intelligent agent under dynamic traffic conditions, and Optimization of traffic flow with RL algorithms.
For further work, research can encompass adjusting the form of the reward signal to better guide the agent’s learning or altering reward shaping strategies to optimize performance. Additionally, investigating the impact of changing the encoding method for traffic signal states (e.g., one-hot encoding) on the agent’s performance would be an important avenue of study.
1
