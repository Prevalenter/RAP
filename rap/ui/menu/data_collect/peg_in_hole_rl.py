import numpy as np
import time
from PyQt5.QtCore import QTimer
from gym import spaces
from gym.utils import seeding
from utils import utils
import TD3
import torch
import gym
import argparse
import random
import os
import sys
sys.path.append('..')

from PyQt5.QtCore import QThread

class PegInHoleTrain(QThread):
    def __init__(self, up_ctrl=None, parent=None):
        super(PegInHoleTrain, self).__init__()

        self.up_ctrl = up_ctrl
        self.parent = parent

        self.env = PegInHoleRL(up_ctrl, parent)

    def run(self):
        self.env.reset()
        print('init down')

        for step in range(int(2e4)):
            self.env.step(None)


class PegInHoleRL:
    def __init__(self, up_ctrl=None, parent=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.parent = parent
        self.dt = dt
        self.assemble_stage_flage = 0


        self.timesteps = 0

        self.max_episodes = 400

        action_high = np.array([0.0005, 0.0005])
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        state_high = np.array([0.6, 0.55])
        state_low = np.array([0.3, 0.25])

        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)

        self.xy_tgt = np.array( [0.4776, 0.3969] )

        self.F_r = np.array([0, 0, 5, 0, 0, 0])
        self.P = np.array([0, 0, 1e-4, 0, 0, 0])
        self.I = np.array([0, 0, 0, 0, 0, 0])
        self.D = np.array([0, 0, 0, 0, 0, 0])

        self.error_sum = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
        self.error_last = np.array([0, 0, 0, 0, 0, 0])

        self.prepared = False

        if up_ctrl is not None:
            print('PegInHoleRL init')
            self.timer_stage12 = QTimer(self.up_ctrl)
            self.timer_stage12.timeout.connect(self.run_stage12)

        self.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()
        self.time_rl = 0

        self.D_rl = 0.02
        self.d_reach = 0.001


    def set_label_ctrl_state(self, label_ctrl_state):
        self.label_ctrl_state = label_ctrl_state


    def run_stage12(self):

        print('run_stage12 run')

        # is_contraol = True
        # if self.up_ctrl is not None and is_contraol:
        force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
        force_contact_norm = np.linalg.norm(force_contact_world[:3])
        torque_norm = np.linalg.norm(force_contact_world[3:])

        if force_contact_norm > 40 or torque_norm > 0.5:
            return

        # only reaction to the force in z positive
        if force_contact_world[2] < 1:
            force_contact_world[2] = 0

        error = force_contact_world - self.F_r
        self.error_sum += error

        dx = np.zeros(6)

        # no contact
        if self.assemble_stage_flage == 0:
            if force_contact_norm == 0:  # no conatact
                dx = np.array([0, 0, -8e-5, 0, 0, 0])
            else:
                self.assemble_stage_flage = 1
                # self.label_ctrl_state.setText(f"Control: stage 1")

        if self.assemble_stage_flage == 1:  # reach the force desired
            dx = np.zeros(6)
            dx[2] = 5e-6 * np.sign(error[2])

            if np.abs(error[2]) < 1:
                self.assemble_stage_flage = 2
                # # self.label_ctrl_state.setText(f"Control: stage 2")
                #
                self.timer_stage12.stop()

        self.x_r += dx
        self.up_ctrl.connect_widget.apply_Rot(self.x_r)


    def reset(self):

        self.prepared = False

        self.parent.on_initial_position()
        time.sleep(3)

        # random state
        self.parent.on_random_position()
        time.sleep(3)

        self.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()

        print('run_stage12 run')

        assemble_stage_flage = 0

        while self.assemble_stage_flage!=2:
            print('self.assemble_stage_flage: ', self.assemble_stage_flage)

            # is_contraol = True
            # if self.up_ctrl is not None and is_contraol:
            force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
            force_contact_norm = np.linalg.norm(force_contact_world[:3])
            torque_norm = np.linalg.norm(force_contact_world[3:])

            if force_contact_norm > 40 or torque_norm > 0.5:
                return

            # only reaction to the force in z positive
            if force_contact_world[2] < 1:
                force_contact_world[2] = 0

            error = force_contact_world - self.F_r
            self.error_sum += error

            dx = np.zeros(6)

            # no contact
            if self.assemble_stage_flage == 0:
                if force_contact_norm == 0:  # no conatact
                    dx = np.array([0, 0, -8e-5, 0, 0, 0])
                else:
                    self.assemble_stage_flage = 1
                    # self.label_ctrl_state.setText(f"Control: stage 1")

            if self.assemble_stage_flage == 1:  # reach the force desired
                dx = np.zeros(6)
                dx[2] = 5e-6 * np.sign(error[2])

                if np.abs(error[2]) < 1:
                    self.assemble_stage_flage = 2
                    # # self.label_ctrl_state.setText(f"Control: stage 2")

            self.x_r += dx
            self.up_ctrl.connect_widget.apply_Rot(self.x_r)

            time.sleep(0.1)

        print('reset down!')


        # the state include the xy and force sensor
        state = {}

        return state


    def step(self, action):

        force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
        force_contact_norm = np.linalg.norm(force_contact_world[:3])
        torque_norm = np.linalg.norm(force_contact_world[3:])

        if force_contact_norm > 40 or torque_norm > 0.5:
            return

        # only reaction to the force in z positive
        if force_contact_world[2] < 1:
            force_contact_world[2] = 0

        error = force_contact_world - self.F_r
        self.error_sum += error

        dx = np.zeros(6)


        if self.assemble_stage_flage == 2:  # to the hole
            dx = np.zeros(6)

            dx[:2] = -(8e-5) * np.sign(self.x_r[:2] - self.xy_tgt[:2])

            # keep the force in z axis constant
            # dx[2] = 5e-6 * np.sign(error[2])
            if np.abs(error[2]) > 0.5:
                dx[2] = 5e-6 * np.sign(error[2])

            if self.x_r[2] < 0.172 or force_contact_norm < 2:
                self.assemble_stage_flage = 3
                # self.label_ctrl_state.setText(f"Control: stage 3")

        # if self.assemble_stage_flage == 3:
        #     dx = np.zeros(6)
        #     if np.abs(error[0]) > 0.5:
        #         dx[0] = 2e-6 * np.sign(error[0])
        #     if np.abs(error[0]) > 0.5:
        #         dx[1] = 2e-6 * np.sign(error[1])
        #
        #     if np.abs(error[2]) > 0.5:
        #         dx[2] = 5e-5 * np.sign(error[2])
        #
        #     if np.abs(error[3]) > 0.2:
        #         dx[3] = 1e-4 * np.sign(error[3])
        #
        #     if np.abs(error[4]) > 0.2:
        #         dx[4] = 1e-4 * np.sign(error[4])
        #
        #     if self.x_r[2] < 0.120:
        #         self.assemble_stage_flage = 4
        #         self.label_ctrl_state.setText(f"Control: stage 4")

        self.error_last = error

        self.x_r += dx
        #
        self.up_ctrl.connect_widget.apply_Rot(self.x_r)

        # print('step')
        time.sleep(0.1)

    def reaction_foce(self, pos):
        if pos[2]>0.55:
            return np.zeros(6)
        else:
            dx = 0.55-np.round(pos[2], 3)
            # print(dx)
            return np.array([0, 0, np.exp(100*dx)+3, 0, 0, 0])


    def env_pre(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="Gear1-v0")  # Gear1 Robot_axis6
        parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=10e3, type=int)  # Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=5e5, type=int)  # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.05)  # Std of Gaussian exploration noise
        parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.95)  # Discount factor
        parser.add_argument("--tau", default=0.005)  # Target network update rate
        parser.add_argument("--policy_noise", default=0.1)  # 0.2 Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
        parser.add_argument("--save_model", default=True, action="store_true")  # Save model and optimizer parameters
        parser.add_argument("--load_model",
                            default="")  # Model load file name, "" doesn't load, "default" uses file_name
        self.args = parser.parse_args()

        self.file_name = f"{self.args.policy}_{self.args.env}_{self.args.seed}"

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if self.args.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        self.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)

        self.action_space.seed(self.args.seed)
        self.observation_space.seed(self.args.seed)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.max_action = float(self.action_space.high[0])

        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "discount": self.args.discount,
            "tau": self.args.tau,
        }

        if self.args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = self.args.policy_noise * self.max_action
            kwargs["noise_clip"] = self.args.noise_clip * self.max_action
            kwargs["policy_freq"] = self.args.policy_freq
            kwargs["seed"] = self.args.seed
            self.policy = TD3.TD3(**kwargs)

        self.replay_buffer = utils.ReplayBuffer(self.state_dim, self.action_dim)

        self.evaluations = []

        self.reward_list = []

        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def eval_policy(self , policy, env_name, seed, timesteps, eval_episodes=5):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)
        eval_env.timesteps = timesteps

        policy.actor.eval()

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        policy.actor.train()

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward