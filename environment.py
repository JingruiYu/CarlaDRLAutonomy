from carla_tools import *

import time
import gym
from gym import spaces




class environment(object):
    def __init__(self,args,model='sac'):
        self.targ_e = 40
        self.model = model
        self.colDic = ['Pole','Unknown'] 
        self.typDic = ['Sidewalk','Pole','Unknown','Road']

        pygame.init()
        pygame.font.init()
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        self.display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        self.world = World(client.get_world(), hud, args)
        
        self.laststate = None

        if self.model == 'dqn':
            self.actDic = np.mat(' \
                0.2, 0.0, 0.0; \
                0.4, 0.0, 0.0; \
                0.6, 0.0, 0.0; \
                0.8, 0.0, 0.0; \
                0.2,-0.2, 0.0; \
                0.2, 0.2, 0.0; \
                0.2,-0.4, 0.0; \
                0.2, 0.4, 0.0; \
                0.4,-0.2, 0.0; \
                0.4, 0.2, 0.0; \
                0.0, 0.0, 1.0')
            self.action_space = spaces.Discrete(len(self.actDic))
        elif self.model == 'sac':
            self.action_bound = [-1, 1]
            self.action_dim = 3
            self.action_space= spaces.Box(low=-1,high=1,shape=[self.action_dim])

        self.min_x = -11.0
        self.max_x = 0.1
        self.min_y = -120.0
        self.max_y = -20.0
        # self.min_z = -0.1
        # self.max_z = 6.29
        self.min_vx = -0.1
        self.max_vx = 11.0
        self.min_vy = -0.1
        self.max_vy = 11.0
        # self.min_vz = -20.0
        # self.max_vz = 20.0

        self.low_state = np.array(
            [self.min_x, self.min_y, self.min_vx, self.min_vy], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x, self.max_y, self.max_vx, self.max_vy], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        # self.render()
        self.reset()
        self.world.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.stepNum = 0
        self.stopTime = 0

    def step(self, action):
        self.stepNum = self.stepNum + 1

        if self.model == 'dqn':
            control = self.getActionDQN(actionID=action)
        elif self.model == 'sac':
            control = self.getActionSAC(actions=action)
        
        self.world.player.apply_control(control)
        time.sleep(0.2)

        self.state = self.getState()

        self.isSuccess()

        finish = self.isFinish()

        reward = self.getReward(action)
        print('step: ', self.stepNum,' reward:',reward, ' state:',self.state[0],' ',self.state[1],' ',self.state[2],' ',self.state[3] )

        self.laststate = self.state

        return self.state, reward, finish, {'success':self.success}

    def reset(self):
        self.world.restart()
        print('RESET!\n\n')

        self.stepNum = 0
        self.stopTime = 0
        
        self.state = self.getState()
        
        return self.state

    def render(self):
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)

            self.world.tick(clock)
            self.world.render(self.display)
            pygame.display.flip()

            self.step(14)
     
            # if self.world is not None:
            #     self.world.destroy()

        pygame.quit()

    def getReward(self, actions):
        nowdis = np.linalg.norm(np.array([self.state[0], self.state[1]]))
        
        reward = 100 * math.exp(-1*math.sqrt(nowdis)/(2*math.sqrt(9))) - 1.5*nowdis + 10000 * self.success - 20000 * self.collision

        return reward

    def isFinish(self):
        self.collision = False
        colTpye = self.world.collision_sensor.collisionType
        
        if colTpye in self.colDic:
            self.collision = True

        if colTpye is not None and colTpye not in self.typDic:
            print(colTpye)

        stop = 1
        if self.laststate is not None:
            stop = np.linalg.norm(np.array([self.state[0] - self.laststate[0], self.state[1] - self.laststate[1]]))

        if stop < 1e-2:
            self.stopTime = self.stopTime + 1
        else:
            self.stopTime = 0

        if colTpye == 'Sidewalk' and self.stopTime > 3:
            self.collision = True

        if self.stopTime > 50 and not self.success:
            self.collision = True

        lx = self.world.player.get_transform().location.x
        ly = self.world.player.get_transform().location.y

        if (lx > self.max_x):
            self.collision = True
        
        if (lx < self.min_x):
            self.collision = True
        
        if (ly > self.max_y or ly < self.min_y):
            self.collision = True
        
        return self.collision


    def isSuccess(self):
        
        self.success = bool(
            abs(self.state[0]) < 2
            and abs(self.state[1]) < 2
        )
    
    def getState(self):
        #tmp_state = np.array([self.world.player.get_transform().location.x, self.world.player.get_transform().location.y])
        # t_yaw = self.ang2rot(self.world.target_point.rotation.yaw)
        # n_yaw = self.ang2rot(self.world.player.get_transform().rotation.yaw)
        
        target_tranform = [
            self.world.target_point.location.x - self.world.player.get_transform().location.x,
            self.world.target_point.location.y - self.world.player.get_transform().location.y]
            # t_yaw - n_yaw]
        
        # angular_velocity = self.world.player.get_angular_velocity()
        velocity_world = self.world.player.get_velocity()

        tmp_state = np.array([target_tranform[0], target_tranform[1], velocity_world.x, velocity_world.y])
        
        # tmp_state = np.array([target_tranform[0], target_tranform[1], target_tranform[2], velocity_world.x, velocity_world.y, angular_velocity.z])
        return tmp_state

    def getActionDQN(self, actionID=4):
        
        if actionID > 10 or actionID < 0:
            print(actionID)

        self.control = carla.VehicleControl(
            throttle=self.actDic[actionID,0],
            steer=self.actDic[actionID,1],
            brake=self.actDic[actionID,2],
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return self.control

    def getActionSAC(self, actions):
        a_t = (actions[0] + 1) * 0.5
        a_s = actions[1] * 1.0
        a_b = 0.0
        if actions[2] > 0.8:
            a_b = 1.0

        self.control = carla.VehicleControl(
            throttle=a_t,
            steer=a_s,
            brake=a_b,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return self.control

    def get_obsdim(self):
        return 4

    def ang2rot(self,ego_yaw):
        if ego_yaw < 0:
            ego_yaw += 360
        if ego_yaw > 360:
            ego_yaw -= 360
        ego_yaw = ego_yaw/180.0 * 3.141592653
        
        return ego_yaw
# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def getArgs():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='192.168.0.29',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    return args