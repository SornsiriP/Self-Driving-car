from cmath import cos, sin
import gym
import numpy as np
import math
import pybullet as p
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()
        #connect to pybullet
        self.client = p.connect(p.GUI)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.prev_pos = (0,0)
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.current_step = 0

    #Normalize action
    def map_action(self, action):
        speed_range = [0,1]
        steer_range = [-0.6,0.6]
        new_speed = np.interp(action[0],[-1,1],speed_range)
        new_steer = np.interp(action[0],[-1,1],steer_range)
        return [new_speed, new_steer]

    def step(self, action):
        action = self.map_action(action)
        self.current_step +=1
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()
        #Find distance between car and goal
        previous_dist_to_goal = np.linalg.norm(tuple(map(lambda i, j: i - j, self.goal, self.prev_pos)))
        current_dist_to_goal =  np.linalg.norm(tuple(map(lambda i, j: i - j, self.goal, car_ob[0:2])))
        #Reward will be plus if the car is driving to the goal
        reward = previous_dist_to_goal - current_dist_to_goal
        #Collect car position for next step
        self.prev_pos = car_ob[0:2]

        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            self.current_step = 0
            self.done = True
        # Done by reaching goal
        elif current_dist_to_goal < 1:
            self.done = True
            self.current_step = 0
        #Done by doing more than 1000 steps
        elif self.current_step >1000:
            self.current_step = 0
            self.done = True

        # print("reward" ,reward)
        ob = np.array(car_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client)

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()
        return np.array(car_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        plt.pause(.00001)

        return frame[:,:,:3]

    def close(self):
        p.disconnect(self.client)
