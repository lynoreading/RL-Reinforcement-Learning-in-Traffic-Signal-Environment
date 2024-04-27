from dataclasses import field
import pandas as pd
import numpy as np
import gym
from gym import spaces
import numpy as np

'''
The traffic environment we have established is discrete. In this environment, only the lanes from 
east to west and from north to south are considered, and a total of 4 lanes have been optimized. 
Each action corresponds to the green light of each road, and there is a linear relationship between 
the CO2 emissions of the cars and their waiting time. At each step, a random number of vehicles within 
a fixed range is first added to each lane, and then an action is applied to reduce the number of vehicles 
on the lane corresponding to the green light.
'''

class TrafficEnvironment(gym.Env):
    def __init__(self, 
                 num_phases, 
                 out_put_name, 
                 simulation_step=50000, 
                 num_lanes=4, 
                 fixed=False, 
                 seed=None):
        """
        RL env for Traffic Signal

        Args:
            num_phases (int): the number of the phases of traffic signal
            out_put_name (str): info output file name/path 
            simulation_step (int, optional): the step of simulation. Defaults to 50000.
            num_lanes (int, optional): the number of lanes. Defaults to 4.
            fixed (bool, optional): the fixed program of the traffic Signal. Defaults to False.
            seed (int, optional): seed. Defaults to None.
        """
        super(TrafficEnvironment, self).__init__()

        # seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.num_phases = num_phases
        self.num_lanes = num_lanes
        self.lane_max_capacity = 100

        # def action space
        self.action_space = spaces.Discrete(num_phases)

        # def observation space
        observation_space_low = np.zeros(3 * (self.num_lanes+1))
        observation_space_high = np.ones(
            3 * (self.num_lanes+1)) * self.lane_max_capacity * 5
        self.observation_space = spaces.Box(low=observation_space_low,
                                            high=observation_space_high,
                                            dtype=np.float32)
        # create lanes
        self.lanes = [Lane(self.lane_max_capacity, self.np_random)
                      for _ in range(num_lanes)]  # init the lanes

        # iteration variable
        self.new_action = 0
        self.last_action = 0
        self.max_time = 5
        self.min_time = 1
        self.time_step = 0
        self.last_action_time = 0
        self.simulation_step = simulation_step
        self.last_measure = 0

        # info output
        self.info_metrics = []
        self.num_epoch = 0

        # output .csv file path
        self.out_put_name = out_put_name

        # fixed
        self.fixed = fixed
        self.fixed_action = [0, 1, 2, 3]
        self.fixed_counter = 0

        # co2 emission types
        self.type1_percentage = 0.7
        self.type2_percentage = 0.3

    # def set_traffic_light_state(self, state):
        # self.traffic_light.set_state(state)

    def reset(self):
        # super().reset(seed=seed)

        if self.num_epoch != 0:
            self.save_csv(self.out_put_name, self.num_epoch)
        self.num_epoch += 1
        self.info_metrics = []

        self.time_step = 0
        self.last_action_time = 0
        for lane in self.lanes:
            lane.clear_vehicles()

        observations = self.compute_observation()
        return observations

    def step(self, action):

        self.time_step += 1

        '''
        if the minimum time requirement is met, then the traffic light can change to next phase
        '''

        if self.time_step - self.last_action_time >= self.min_time:

            '''
            traffic signal is fixed, which means the action will change as follow from 0 -> 1 -> 2 -> 3 -> 0 ....
            '''
            if self.fixed:
                self.fixed_counter += 1
                # self.new_action = np.random.randint(0, 4)
                self.new_action = self.fixed_action[self.fixed_counter %
                                                    self.num_phases]
                self.last_action = self.new_action
                self.last_action_time = self.time_step
            else:
                if self.new_action != action:
                    self.last_action_time = self.time_step
                self.new_action = action
                self.last_action = action

        else:
            self.new_action = self.last_action

        # print("Time since last action:",  self.last_action_time)

        # Update the last action time
        # self.last_action = action
        # self.last_action_time = self.time_step

        '''
        generate the new vehicles for each lane
        at this stage it is generating random 0 to 10vehicles at a time
        and then update the waiting time of all the vehicles in each lane 
        '''

        # for lane in self.lanes:
        #     num_new_vehicles = self.np_random.integers(0, 10)
        #     for _ in range(num_new_vehicles):
        #         vehicle = Vehicle(f"Vehicle_{self.time_step}",self.np_random)
        #         lane.add_vehicle(vehicle)
        #     lane.update_vehicles_waiting_time()

        '''
        add new feature: two types of vehicles
        '''
        for i, lane in enumerate(self.lanes):
            num_new_vehicles = self.np_random.integers(0, 10)

            # Check if the lane index is 0 or 1 (first and second lanes)
            if i in [0, 1]:
                num_type1_vehicles = int(
                    num_new_vehicles * self.type1_percentage)
                num_type2_vehicles = num_new_vehicles - num_type1_vehicles
            else:
                # For lanes 2 and 3, the percentages are reversed
                num_type2_vehicles = int(
                    num_new_vehicles * self.type1_percentage)
                num_type1_vehicles = num_new_vehicles - num_type2_vehicles

            # Generate type1 vehicles
            for _ in range(num_type1_vehicles):
                vehicle = Vehicle(
                    f"Vehicle_{self.time_step}", self.np_random, "type1")
                lane.add_vehicle(vehicle)

            # Generate type2 vehicles
            for _ in range(num_type2_vehicles):
                vehicle = Vehicle(
                    f"Vehicle_{self.time_step}", self.np_random, "type2")
                lane.add_vehicle(vehicle)

            lane.update_vehicles_waiting_time()

        """
        INTERSECTION VIEW
                               N
                          ↓ Lane2  go straight or right
                            ↓ lane3   go left
                        |      |      |
                        |      |      |        
                        |      |      | 
                        |      |      | 
                        |      |      | 
                        |      |      |                 
        ----------------               ----------------
                                                ←   lane0  go straight or right
                            traffic             ←   lane1  go left
    W   ----------------               ----------------  E
                             signal
                                          
        ----------------               ----------------
                        |      |      |
                        |      |      |        
                        |      |      | 
                        |      |      | 
                        |      |      | 
                        |      |      |          

                               S
        """

        ''' apply the action'''

        if self.new_action == 0:  # w/e->g1 n/s->red
            self.lanes[0].remove_vehicle()
        elif self.new_action == 1:  # w/e->g2 turn left n/s->red
            self.lanes[1].remove_vehicle()
        elif self.new_action == 2:  # w/e->red n/s->g1
            self.lanes[2].remove_vehicle()
        elif self.new_action == 3:  # w/e->red n/s->g2 turn left
            self.lanes[3].remove_vehicle()

        # # ! Debug for env
        # debug_info = self._simlulation_debug_info(self.new_action)
        # self.info_metrics.append(debug_info)

        if self.time_step % 5 == 0:
            info = self.compute_step_info()
            self.info_metrics.append(info)

        if self.time_step % 200 == 0:
            print(f" \rStep: {self.time_step}%", end="")

        '''
        get reward
        '''
        # reward = self._queue_reward()
        reward = self.compute_reward()
        # print("reward:", reward, "action:", new_action, "step:", self.time_step,"action:",  self.last_action,"action:",  action)

        '''
        get terminal signal
        '''
        done = self.time_step >= self.simulation_step

        '''
        get observation
        '''
        observations = self.compute_observation()

        return observations, reward, done, {}

    def save_csv(self, out_csv_name, epoh):
        if out_csv_name is not None:
            df = pd.DataFrame(self.info_metrics)
            df.to_csv(out_csv_name + '_run{}'.format(epoh) +
                      '.csv', index=False)

    def compute_step_info(self):
        return {
            'sim_step': self.time_step,
            'reward': self._queue_reward(),
            # 'lane0_queue': self.lanes[0].num_vehicles(),
            # 'lane1_queue': self.lanes[1].num_vehicles(),
            # 'lane2_queue': self.lanes[2].num_vehicles(),
            # 'lane3_queue': self.lanes[3].num_vehicles(),
            'total_queue': sum(lane.num_vehicles() for lane in self.lanes),
            'total_waiting_time': sum(lane.get_total_waiting_time() for lane in self.lanes),
            'total_co2_emissions': sum(lane.get_total_co2_emissions() for lane in self.lanes)
        }

    def compute_observation(self):  # size: (4+1) * 3
        # queue - number of vehicles
        num_vehicles = [lane.num_vehicles() for lane in self.lanes]
        observation = num_vehicles
        observation.append(sum(num_vehicles))
        # waiting time of vehicles
        waiting_time = [lane.get_total_waiting_time() for lane in self.lanes]
        observation.extend(waiting_time)
        observation.append(sum(waiting_time))
        # co2 emission
        emission = [lane.get_total_co2_emissions() for lane in self.lanes]
        observation.extend(emission)
        observation.append(sum(emission))
        # observation.append(self.new_action)
        observation = np.array(observation)
        return observation

    def get_state(self):
        # get current state - the number of car on each lanes
        # state = [lane.num_vehicles() for lane in self.lanes]
        state = sum(lane.num_vehicles() for lane in self.lanes)
        return state

    ''' chosse reward function'''

    def compute_reward(self):
        # reward = self._queue_reward()
        # reward = self._queue_average_reward()
        # reward = self._waiting_time_reward()
        # reward = self._waiting_time_reward2()
        reward = self._waiting_time_reward3()
        # reward = self._co2_reward()
        return reward

    def _queue_reward(self):
        """
        Punishing the increase in the number of parked vehicles directly.
        """
        return - (self.get_state())**2

    def _queue_average_reward(self):
        new_average = np.mean(self.get_state())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _waiting_time_reward(self):
        ts_wait = sum(lane.get_total_waiting_time()
                      for lane in self.lanes) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(lane.get_total_waiting_time()
                      for lane in self.lanes) / 100.0
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        # ts_wait = sum(self.get_waiting_time())
        # change by me
        ts_wait = sum(lane.get_total_waiting_time()
                      for lane in self.lanes) / 100.0
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def _co2_reward(self):
        alpha = 0  # weighting
        vehicle_base_co2 = 1
        new_co2 = sum(lane.get_total_co2_emissions()
                      for lane in self.lanes)
        reward = self.last_measure - new_co2
        self.last_measure = new_co2
        return reward

    ######################################
    # ! for debug

    def _simlulation_debug_info(self, action):
        return {
            'sim_step': self.time_step,
            'action': action,
            'lane0_queue': self.lanes[0].num_vehicles(),
            'lane1_queue': self.lanes[1].num_vehicles(),
            'lane2_queue': self.lanes[2].num_vehicles(),
            'lane3_queue': self.lanes[3].num_vehicles(),
            'lane0_vehicles_waiting_time': self.lanes[0].vehicles_waiting_time_info(),
            'lane1_vehicles_waiting_time': self.lanes[1].vehicles_waiting_time_info(),
            'lane2_vehicles_waiting_time': self.lanes[2].vehicles_waiting_time_info(),
            'lane3_vehicles_waiting_time': self.lanes[3].vehicles_waiting_time_info(),
            'lane0_vehicles_cow_emissions': self.lanes[0].vehicles_co2_emissions_info(),
            'lane1_vehicles_cow_emissions': self.lanes[1].vehicles_co2_emissions_info(),
            'lane2_vehicles_cow_emissions': self.lanes[2].vehicles_co2_emissions_info(),
            'lane3_vehicles_cow_emissions': self.lanes[3].vehicles_co2_emissions_info()
        }
    ######################################

    def _queue_reward(self):
        return - (self.get_state())**2

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class Lane:

    def __init__(self, max_capacity, seed):
        """
        The Lane class represents a single lane in a traffic simulation environment,
        designed to manage the vehicles within it. Each lane has a maximum capacity,
        indicating the number of vehicles it can accommodate at any given time.

        Args:
            max_capacity (int): the maximum number of vehicles on lane
            seed (RandomState): seed 
        """

        self.max_capacity = max_capacity
        self.np_random = seed
        self.vehicles = []  # vehicles list contains the elements, each represent as a vehicles class, wich includes id, waiting time, and co2 emission

    # ! could be out of the capacity
    def add_vehicle(self, vehicle):
        if len(self.vehicles) < self.max_capacity:
            self.vehicles.append(vehicle)

    def remove_vehicle(self):
        if self.vehicles:  # if not empty
            ''' assume that each step 18 to 22 vehicles can leave the lane'''
            num_removed = self.np_random.integers(18, 22)
            removed_vehicles = self.vehicles[:num_removed]
            self.vehicles = self.vehicles[num_removed:]

    def clear_vehicles(self):
        self.vehicles = []

    def num_vehicles(self):
        return len(self.vehicles)

    def update_vehicles_waiting_time(self):
        for vehicle in self.vehicles:
            vehicle.update_waiting_time_weight()

    def get_total_waiting_time(self):
        total_waiting_time = 0
        for vehicle in self.vehicles:
            total_waiting_time += vehicle.get_waiting_time()
        return total_waiting_time

    def get_total_co2_emissions(self):
        total_co2_emission2 = 0
        for vehicle in self.vehicles:
            total_co2_emission2 += vehicle.get_co2_emissions()
        return total_co2_emission2

    ######################################
    # ! for debug

    def vehicles_waiting_time_info(self):
        info = ""
        for vehlicle in self.vehicles:
            vehlicle_info = str(vehlicle.id) + "-" + \
                str(vehlicle.get_waiting_time()) + "|"
            info += vehlicle_info
        return info

    def vehicles_co2_emissions_info(self):
        info = ""
        for vehlicle in self.vehicles:
            vehlicle_info = str(vehlicle.id) + "-" + \
                str(vehlicle.get_co2_emissions()) + "|"
            info += vehlicle_info
        return info
    ######################################


class Vehicle:

    def __init__(self, Id, seed, vehicle_type):
        """
        The Vehicle class represents an individual vehicle in a traffic simulation environment,
        tracking its unique identifier, waiting time, CO2 emissions, and type. The class
        allows for the creation of vehicles with different emission rates based on their type,
        simulating real-world differences between vehicle emissions.

        Args:
            Id (str): Vehicle ID in form Vehicle_{self.time_step}
            seed (RandomState): seed
            vehicle_type (str): The type of the vehicle, affecting its CO2 emission rate.
        Raises:
            ValueError: unknown vehicle type
        """

        self.id = Id
        self.np_random = seed
        self.waiting_time_weight = 0
        self.vehicle_type = vehicle_type
        # self.co2_emission_per_steps = self.np_random.integers(50, 101) * 0.1
        if self.vehicle_type == "type1":
            self.co2_emission_per_steps = 50 * 0.1
        elif self.vehicle_type == "type2":
            self.co2_emission_per_steps = 100 * 0.1
        else:
            raise ValueError("Invalid vehicle type")

    def update_waiting_time_weight(self):
        self.waiting_time_weight += 1

    def get_waiting_time(self):
        return self.waiting_time_weight

    def get_co2_emissions(self):
        ''' 
        assume that the CO2 emissions from automobiles follow a linear trend:
        emissionOfCo2 = k(co2_emission_per_steps) * time + bias
        '''
        bias = self.np_random.integers(10, 21) * 0.1
        return self.waiting_time_weight * self.co2_emission_per_steps + bias
