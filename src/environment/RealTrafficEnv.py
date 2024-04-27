import pandas as pd
import numpy as np
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

'''
In this environment, a simulation of a real traffic flow scenario at a 
crossroads intersection over a day (86,400 seconds, with one step representing 
one second) is conducted . The data includes the total hourly traffic volume 
on one lane for all 24 hours of the day, which allows for the calculation of 
the probability of a car appearing on the road every second. During green light phases, 
it is estimated that 0.45 vehicles can leave the lane per second, which means approximately 
13 or 14 vehicles can depart in half a minute.

'''


class RealTrafficEnvironment(gym.Env):
    def __init__(self, num_phases, out_put_name, simulation_step=50000, num_lanes=8, fixed=False, seed=None):
        super(RealTrafficEnvironment, self).__init__()
        # seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.num_phases = num_phases
        self.num_lanes = num_lanes
        # self.current_state = None
        self.lane_max_capacity = 200

        # def action space
        self.action_space = spaces.Discrete(num_phases)

        '''def observation space (queue + waiting time + co2 emission)'''
        # observation_space_low = np.zeros(3 * (self.num_lanes+1))
        # observation_space_high = np.ones(
        #     3 * (self.num_lanes+1)) * self.lane_max_capacity * 5
        # self.observation_space = spaces.Box(low=observation_space_low,
        #                                     high=observation_space_high,
        #                                     dtype=np.float32)

        '''def observation space (queue)'''
        observation_space_low = np.zeros(self.num_lanes+1)
        observation_space_high = np.ones(
            self.num_lanes+1) * np.inf
        self.observation_space = spaces.Box(low=observation_space_low,
                                            high=observation_space_high,
                                            dtype=np.float32)

        self.lanes = [Lane(self.lane_max_capacity, self.np_random)
                      for _ in range(num_lanes)]  # init the lanes

        self.new_action = 0
        self.last_action = 0
        # self.max_time = 50
        self.min_time = 30
        self.time_step = 0
        self.last_action_time = 0
        self.simulation_step = simulation_step
        self.last_measure = 0

        self.info_metrics = []
        self.num_epoch = 0

        self.out_put_name = out_put_name

        self.fixed = fixed
        self.fixed_action = [0, 1, 2, 3]
        self.fixed_counter = 0

################################################ PROBABLITY ################################################

        '''
        new vehicles generator acrroding to the traffic flow in a day
        
        SegmentID ,Roadway Name  ,From          ,To               ,Direction ,Date       ,12:00-1:00 AM ,1:00-2:00AM ,2:00-3:00AM ,3:00-4:00AM ,4:00-5:00AM ,5:00-6:00AM ,
        22921 ,     3 AVENUE     ,DEAN STREET   ,PACIFIC STREET   ,NB        ,10/28/2012 ,          211 ,        202 ,        173 ,        145 ,        190 ,        126 ,        
        22921 ,     3 AVENUE     ,DEAN STREET   ,PACIFIC STREET   ,SB        ,10/28/2012 ,           98 ,         92 ,         61 ,         57 ,         53 ,         45 ,        
        
        6:00-7:00AM ,7:00-8:00AM ,8:00-9:00AM ,9:00-10:00AM ,10:00-11:00AM ,11:00-12:00PM ,12:00-1:00PM ,1:00-2:00PM ,2:00-3:00PM ,3:00-4:00PM ,4:00-5:00PM ,5:00-6:00PM ,
        123 ,        139 ,        198 ,         255 ,          273 ,          313 ,         363 ,        385 ,        385 ,        395 ,        339 ,        307 ,        
         25 ,         45 ,         51 ,          35 ,           80 ,           92 ,         133 ,        143 ,        145 ,        142 ,        135 ,        145 ,         
        
        6:00-7:00PM ,7:00-8:00PM ,8:00-9:00PM ,9:00-10:00PM ,10:00-11:00PM ,11:00-12:00AM ,day_total ,day_mean
        249 ,        198 ,        130 ,         106 ,           79 ,           80 ,     5364 ,223.5
        99 ,         72 ,         43 ,          51 ,           35 ,           33 ,     1910 , 79.58333333333333
        '''

        self.traffic_flow_data_nb = [211, 202, 173, 145, 190, 126, 123, 139, 198,
                                     255, 273, 313, 363, 385, 385, 395, 339, 307, 249, 198, 130, 106, 79, 80]
        self.traffic_flow_data_sb = [98, 92, 61, 57, 53, 45, 25, 45, 51, 35,
                                     80, 92, 133, 143, 145, 142, 135, 145, 99, 72, 43, 51, 35, 33, ]
        self.prob_vehicles_per_step_nb = [
            x / 3600 for x in self.traffic_flow_data_nb]
        self.prob_vehicles_per_step_sb = [
            x / 3600 for x in self.traffic_flow_data_sb]

        '''
        SegmentID ,Roadway Name  ,From            ,To               ,Direction ,Date       ,12:00-1:00 AM ,1:00-2:00AM ,2:00-3:00AM ,3:00-4:00AM ,4:00-5:00AM ,5:00-6:00AM ,
        31320     ,PARK AVENUE   ,TOMPKINS AVENUE ,DELMONICO PLACE ,WB         ,01/09/2012 ,           52 ,         18 ,         13 ,         20 ,         20 ,         49 ,        
        31320     ,PARK AVENUE   ,TOMPKINS AVENUE ,DELMONICO PLACE ,EB         ,01/09/2012 ,           59 ,         17 ,         13 ,         21 ,         17 ,         41 ,
        
        6:00-7:00AM ,7:00-8:00AM ,8:00-9:00AM ,9:00-10:00AM ,10:00-11:00AM ,11:00-12:00PM ,12:00-1:00PM ,1:00-2:00PM ,2:00-3:00PM ,3:00-4:00PM ,4:00-5:00PM ,5:00-6:00PM ,
                111 ,        279 ,        315 ,         208 ,          207 ,          202 ,         223 ,        127 ,        205 ,        229 ,        239 ,        243 , 
                 85 ,        221 ,        256 ,         199 ,          111 ,          169 ,         174 ,        211 ,        191 ,        302 ,        413 ,        311 ,
        
        6:00-7:00PM ,7:00-8:00PM ,8:00-9:00PM ,9:00-10:00PM ,10:00-11:00PM ,11:00-12:00AM ,day_total ,day_mean
                211 ,        122 ,        113 ,         108 ,           84 ,           63 ,     3461 ,144.20833333333334
                258 ,        148 ,        106 ,         104 ,           82 ,           85 ,     3594 ,149.75
        '''
        self.traffic_flow_data_wb = [52, 18, 13, 20, 20, 49, 111, 279, 315, 208,
                                     207, 202, 223, 127, 205, 229, 239, 243, 211, 122, 113, 108, 84, 63]
        self.traffic_flow_data_eb = [59, 17, 13, 21, 17, 41, 85, 221, 256, 199,
                                     111, 169, 174, 211, 191, 302, 413, 311, 258, 148, 106, 104, 82, 85]
        self.prob_vehicles_per_step_wb = [
            x / 3600 for x in self.traffic_flow_data_wb]
        self.prob_vehicles_per_step_eb = [
            x / 3600 for x in self.traffic_flow_data_eb]

################################################################################################################################################

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

        # ? observation also need to return phase?
        # ? (phase, min_green, ... laneDensity, ...laneQueue)
        observations = self.compute_observation()
        return observations

    def step(self, action):
        """
        Gym env api - step

        Parameters:
        - action (int)

        Returns:
        - state (object):
        - reward (float): 
        - done (bool): 
        - info (dict): 
        """
        # ? should return the observation not next state

        self.time_step += 1

        # Check if the time interval since the last action is within the allowed range
        # the traffic signal could changed to next phase
        if self.time_step - self.last_action_time >= self.min_time:

            '''
            traffic signal is fixed, which means the action will change as follow from 0 -> 1 -> 2 -> 3 -> 0 ....
            '''
            if self.fixed:
                self.fixed_counter += 1
                # self.new_action = np.random.integers(0, 4)
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
        at this stage it is generating random 0 to 5 vehicles at a time
        and then update the waiting time of all the vehicles in each lane 
        '''

        '''get the probability for generating the vehicle,
            if the random value smaller than the probability, then there is no vehicle will be generated
            if bigger, there will be a vehicle added in to the lane
        '''
        prob_vehicle_nb = self.prob_vehicles_per_step_nb[self.time_step // 3600]
        prob_vehicle_sb = self.prob_vehicles_per_step_sb[self.time_step // 3600]
        prob_vehicle_wb = self.prob_vehicles_per_step_wb[self.time_step // 3600]
        prob_vehicle_eb = self.prob_vehicles_per_step_eb[self.time_step // 3600]

        ''' nb: lane 6/7 '''
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_nb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[6].add_vehicle(vehicle)
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_nb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[7].add_vehicle(vehicle)

        ''' sb: lane 2/3 '''
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_sb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[2].add_vehicle(vehicle)
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_sb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[3].add_vehicle(vehicle)

        ''' wb: lane 0/1 '''
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_wb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[0].add_vehicle(vehicle)
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_wb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[1].add_vehicle(vehicle)

        ''' eb: lane 4/5 '''
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_eb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[4].add_vehicle(vehicle)
        random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
        if random_number <= prob_vehicle_eb:
            vehicle = Vehicle(f"Vehicle_{self.time_step}", self.np_random)
            self.lanes[5].add_vehicle(vehicle)

        ''' update the waiting of each vehicle on lanes'''
        for lane in self.lanes:
            lane.update_vehicles_waiting_time()

        # * generator the random(x, y) vehicles per step
        # for lane in self.lanes:
        #     num_new_vehicles = self.np_random.integers(low=0, high=10)  # ?
        #     for _ in range(num_new_vehicles):
        #         vehicle = Vehicle(f"Vehicle_{self.time_step}")
        #         lane.add_vehicle(vehicle)
        #     lane.update_vehicles_waiting_time()

        """
        INTERSECTION VIEW
                               
 
                                                  N
 
                                              ↓ lane2 go straight / right
                                                 ↓ lane3 go left
                                                
                                            |      |      |
                                            |      |      |        
                                            |      |      | 
                                            |      |      | 
                                            |      |      | 
                                            |      |      |                 
                            ----------------               ----------------
                                                                    ←   lane0  go straight / right
                                                traffic             ←   lane1  go left
                        W   ----------------               ----------------                E
    lane 4  go left →                            signal
    lane 5  go straight / right  →
                            ----------------               ----------------
                                            |      |      |
                                            |      |      |        
                                            |      |      | 
                                            |      |      | 
                                            |      |      | 
                                            |      |      |          
                                                        ↑ lane6 go straight / right
                                                      ↑  lane7 go left
                                                   S
        """

        ''' apply the action'''

        if self.new_action == 0:  # lane0 and lane5 green
            self.lanes[0].remove_vehicle()
            self.lanes[5].remove_vehicle()
        elif self.new_action == 1:  # lane3 and lane7 green
            self.lanes[3].remove_vehicle()
            self.lanes[7].remove_vehicle()
        elif self.new_action == 2:  # lane2 and lane6 green
            self.lanes[2].remove_vehicle()
            self.lanes[6].remove_vehicle()
        elif self.new_action == 3:  # lane4 and lane1 green
            self.lanes[4].remove_vehicle()
            self.lanes[1].remove_vehicle()


##########################################################################################
        # ! Debug for simulation
        # debug_info = self._simlulation_debug_info(self.new_action)
        # self.info_metrics.append(debug_info)

        # # TODO: when action apply is finish, then comupte the info
        if self.time_step % 5 == 0:
            info = self.compute_step_info()
            self.info_metrics.append(info)

        if self.time_step % 200 == 0:
            print(f" \rStep: {self.time_step}%", end="")

##########################################################################################
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

        # ? should return observation, reward, *truncated, INFO, done
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
            'lane0_queue': self.lanes[0].num_vehicles(),
            'lane1_queue': self.lanes[1].num_vehicles(),
            'lane2_queue': self.lanes[2].num_vehicles(),
            'lane3_queue': self.lanes[3].num_vehicles(),
            'lane4_queue': self.lanes[4].num_vehicles(),
            'lane5_queue': self.lanes[5].num_vehicles(),
            'lane6_queue': self.lanes[6].num_vehicles(),
            'lane7_queue': self.lanes[7].num_vehicles(),
            'total_queue': sum(lane.num_vehicles() for lane in self.lanes),

            # 'total_waiting_time': sum(lane.get_total_waiting_time() for lane in self.lanes),
            # 'total_co2_emissions': sum(lane.get_total_co2_emissions() for lane in self.lanes)
        }

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

    def compute_observation(self):  # size: (4+1) * 3
        '''queue - number of vehicles'''
        num_vehicles = [lane.num_vehicles() for lane in self.lanes]
        observation = num_vehicles
        observation.append(sum(num_vehicles))
        '''waiting time of vehicles'''
        # waiting_time = [lane.get_total_waiting_time() for lane in self.lanes]
        # observation.extend(waiting_time)
        # observation.append(sum(waiting_time))
        '''co2 emission'''
        # emission = [lane.get_total_co2_emissions() for lane in self.lanes]
        # observation.extend(emission)
        # observation.append(sum(emission))
        '''? action phase'''
        # observation.append(self.new_action)
        observation = np.array(observation)
        return observation

    def get_state(self):
        # get current state - the number of car on each lanes
        # state = [lane.num_vehicles() for lane in self.lanes]
        state = sum(lane.num_vehicles() for lane in self.lanes)
        return state

    def compute_reward(self):
        reward = self._queue_average_reward()
        # reward = self._waiting_time_reward()
        # reward = self._waiting_time_reward2()
        # reward = self._waiting_time_reward3()
        return reward

    def calculate_reward(self):
        new_average = np.mean(self.get_state())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_average_reward(self):
        new_average = np.mean(self.get_state())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        """
        Punishing the increase in the number of parked vehicles directly.
        """
        return - (self.get_state())**2

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

    # def _co2_queue_reward(self):
    #     queue = self.get_state()
    #     weight = [(traci.lane.getCO2Emission(lane) / self.vehicle_base_co2 /
    #                max(1, traci.lane.getLastStepVehicleNumber(lane))) for lane in self.lanes]
    #     # weighted_queue = [a * (b+1)  for a, b in zip(queue, self.get_lanes_emission_norm(self.lanes))]
    #     weighted_queue = [a * b for a, b in zip(queue, weight)]
    #     # weighted_queue = [a * b  for a, b in zip(queue, self.get_lane_weight(self.lanes))]
    #     return - (sum(weighted_queue))**2

    def _queue_reward(self):
        return - (self.get_state())**2

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class Lane:

    def __init__(self, max_capacity, seed):
        self.max_capacity = max_capacity
        self.np_random = seed
        self.vehicles = []  # vehicles list contains the elements, each represent as a vehicles class, wich includes id, waiting time, and cow e

    # ! could be out of the capacity
    def add_vehicle(self, vehicle):
        if len(self.vehicles) < self.max_capacity:
            self.vehicles.append(vehicle)

    # def remove_vehicle(self):
    #     if self.vehicles:  # if not empty
    #         num_removed = self.np_random.integers(low=0, high=2)  # ?
    #         removed_vehicles = self.vehicles[:num_removed]
    #         self.vehicles = self.vehicles[num_removed:]

    def remove_vehicle(self):
        prob_remove_vehlices = 0.45
        if self.vehicles:  # if not empty
            random_number = self.np_random.uniform(low=0, high=1, size=(1,))[0]
            if random_number <= prob_remove_vehlices:
                self.vehicles = self.vehicles[1:]  # remove one vehicl

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


class Vehicle:
    def __init__(self, Id, seed):
        self.id = Id
        self.np_random = seed
        self.waiting_time_weight = 0
        self.co2_emission_per_steps = self.np_random.integers(
            low=55, high=100) * 0.1

    def update_waiting_time_weight(self):
        self.waiting_time_weight += 1

    def get_waiting_time(self):
        return self.waiting_time_weight

    def get_co2_emissions(self):
        ''' 
        assume that the CO2 emissions from automobiles follow a linear trend:
        emissionOfCo2 = k(co2_emission_per_steps) * time + bias
        '''
        bias = self.np_random.integers(low=10, high=20) * 0.1
        return self.waiting_time_weight * self.co2_emission_per_steps + bias

