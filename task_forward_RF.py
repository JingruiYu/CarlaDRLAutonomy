from __future__ import print_function
import logging
import math
import time
import collections
from collections import deque
import gym
from gym import spaces, logger
try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')
try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')
import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from carla_tools import *

targ_e = 40
step_T_bound = (0.6,1)		# Boundary of throttle values
step_S_bound = (-0.8,0.8)	# Boundary of the steering angle values

class forward_env():
    def __init__(self, throttleSize=4, steerSize=9, vehicleNum=1, model='dqn'):
        log_level = logging.INFO
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=log_level)
        logging.info('listening to server %s:%s', '127.0.0.1', 2000)

        pygame.init()
        pygame.font.init()
        self.vehicleNum = vehicleNum
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(4.0)
        self.display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(1280, 720)
        self.world = World(self.client.get_world(), self.hud, 'vehicle.*', vehicleNum)
        self.state = np.array([self.world.start_point.location.x,self.world.start_point.location.y])
        self.clock = pygame.time.Clock()
        self.control = carla.VehicleControl(
            throttle=0.5,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)
        self.steer_history = deque(maxlen=100)
        self.throttle_history = deque(maxlen=100)

        self.tire_friction_array = np.arange(3, 4.1, 0.1)
        self.mass_array = np.arange(1700, 1910, 50)
        self.ori_physics_control = self.world.player.get_physics_control()
        self.wheel_fl = self.ori_physics_control.wheels[0]
        self.wheel_fr = self.ori_physics_control.wheels[1]
        self.wheel_rl = self.ori_physics_control.wheels[2]
        self.wheel_rr = self.ori_physics_control.wheels[3]

        self.step_T_pool = [step_T_bound[0]]
        self.step_S_pool = [step_S_bound[0]]
        t_step_rate = (step_T_bound[1]- step_T_bound[0])/throttleSize
        s_step_rate = (step_S_bound[1]- step_S_bound[0])/steerSize
        for i in range(throttleSize):
            self.step_T_pool.append(self.step_T_pool[-1]+t_step_rate)
        for i in range(steerSize):
            self.step_S_pool.append(self.step_S_pool[-1]+s_step_rate)
        print(self.step_T_pool)
        print(self.step_S_pool)
        self.tStateNum = len(self.step_T_pool)
        self.sStateNum = len(self.step_S_pool)
        self.action_space = spaces.Discrete(self.tStateNum*self.sStateNum)
        self.min_x = -0.1
        self.min_y = -120.0
        self.max_x = -11.0
        self.max_y = -20.0
        self.low_state = np.array(
            [self.min_x, self.min_y], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x, self.max_y], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.world.world.set_weather(carla.WeatherParameters.ClearNoon)

    def step(self, action):
        self.control = self.getAction(actionID=action) 

        self.world.player.apply_control(self.control)
        self.steer_history.append(self.control.steer)
        self.throttle_history.append(self.control.throttle)
        # time.sleep(0.05)

        self.state = self.getState()

        reward = self.getReward()

        done = self.isFinish()

        return self.state, reward, done, {}

    def reset(self):
        index_friction = np.random.randint(
            0, self.tire_friction_array.shape[0])
        index_mass = np.random.randint(0, self.mass_array.shape[0])
        self.tire_friction = self.tire_friction_array[index_friction]
        self.mass = self.mass_array[index_mass]
        self.wheel_fl.tire_friction = self.tire_friction
        self.wheel_fr.tire_friction = self.tire_friction
        self.wheel_rl.tire_friction = self.tire_friction
        self.wheel_rr.tire_friction = self.tire_friction
        wheels = [self.wheel_fl, self.wheel_fr, self.wheel_rl, self.wheel_rr]
        self.ori_physics_control.wheels = wheels
        self.ori_physics_control.mass = float(self.mass)
        self.world.player.apply_physics_control(self.ori_physics_control)
        time.sleep(0.5)

        velocity_local = [10, 0]  # 5m/s
        angular_velocity = carla.Vector3D()
        ego_yaw = self.world.start_point.rotation.yaw
        ego_yaw = ego_yaw/180.0 * 3.141592653
        transformed_world_velocity = self.velocity_local2world(
            velocity_local, ego_yaw)
        self.world.player.set_transform(self.world.start_point)
        self.world.player.set_velocity(transformed_world_velocity)
        self.world.player.set_angular_velocity(angular_velocity)
        self.world.player.apply_control(carla.VehicleControl())
        self.world.collision_sensor.history = []

        self.steer_history.clear()
        self.throttle_history.clear()

        print('RESET!\n\n')

        self.state = np.array([self.world.start_point.location.x,self.world.start_point.location.y])
        self.steps_beyond_done = None
        return self.state

    def render(self):
        # agent = BehaviorAgent(self.world.player, behavior='normal')
        clock = pygame.time.Clock()

        # while True:
        clock.tick_busy_loop(60)

        # As soon as the server is ready continue!
        # if not self.world.world.wait_for_tick(10.0):
        #     continue

        # agent.update_information(self.world)

        self.world.tick(clock)
        self.world.render(self.display)
        pygame.display.flip()

        # speed_limit = self.world.player.get_speed_limit()
        # agent.get_local_planner().set_speed(speed_limit)

        # control = self.control
        # self.world.player.apply_control(control)

        if self.world is not None:
            self.world.destroy()

        pygame.quit()

    def get_obsdim(self):
        return 2

    def getReward(self):
        lx = self.state[0]
        ly = self.state[1]
        sx = self.world.start_point.location.x
        sy = self.world.start_point.location.y

        reward = -((lx - sx) + (ly - sy - targ_e))
        return reward

    def getState(self):
        location = self.world.player.get_location()
        state = (location.x,location.y)

        return np.array(state)

    def velocity_local2world(self, velocity_local, yaw):
        vx = velocity_local[0]
        vy = velocity_local[1]

        world_x = vx * math.cos(yaw) - vy * math.sin(yaw)
        world_y = vy * math.cos(yaw) + vx * math.sin(yaw)

        return carla.Vector3D(world_x, world_y, 0)

    def getAction(self, actionID=4):

        throttleID = int(actionID / self.sStateNum)
        steerID = int(actionID % self.sStateNum)

        self.control = carla.VehicleControl(
            throttle=0.5, #self.step_T_pool[throttleID],
            steer=0.0, #self.step_S_pool[steerID],
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return self.control

    def isFinish(self):
        lx = self.state[0]
        ly = self.state[1]
        sx = self.world.start_point.location.x
        sy = self.world.start_point.location.y

        done = bool(
            abs(lx - sx) < 0.1
            and abs(ly - sy - targ_e) < 0.1
        )

        if (lx > -0.1 or lx < -11.0) or ( ly > - 20.0 or ly < -120.0):
            done = True

        return done
