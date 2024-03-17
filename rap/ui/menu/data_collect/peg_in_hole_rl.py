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


    def RL_env_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="Gear1-v0")  # Gear1 Robot_axis6
        parser.add_argument("--seed", default=1000, type=int)  # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=10e3, type=int)  # Time steps initial random policy is used 10e3
        parser.add_argument("--eval_freq", default=2e4, type=int)  # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
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

        self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)

        self.env.action_space.seed(self.args.seed)
        self.env.observation_space.seed(self.args.seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

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



    def run(self):
        state, done = self.env.reset(), False
        print('init down')

        self.RL_env_init()
        replay_buffer = utils.ReplayBuffer(self.state_dim, self.action_dim)
        reward_list = []
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        learn_time=0

        # for step in range(int(2e4)):
        #     next_state, reward, done, action = self.env.step_pid(None)
        #     done_bool = float(done)
        #     replay_buffer.add(state, action, next_state, reward, done_bool)
        #     state = next_state
        #
        #     if done:
        #         reward_list.append(reward)
        #         np.save(f"./results/reward_list_{self.file_name}", reward_list)
        #         state, done = self.env.reset(), False
        #         learn_time += 1
        #         if learn_time >= 10:
        #             break


        for t in range(int(self.args.max_timesteps)):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.args.start_timesteps:

                action = self.env.action_space.sample()
                # if t%300==1: print('action', action)
            else:
                self.policy.actor.eval()
                action = (
                        self.policy.select_action(np.array(state))
                        + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)
                self.policy.actor.train()

            next_state, reward, done = self.env.step_rl(action)
            done_bool = float(done)
            replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state
            episode_reward += reward

            if done:
                reward_list.append(reward)
                np.save(f"./results/reward_list_{self.file_name}", reward_list)

            if t >= self.args.start_timesteps:
                self.policy.train(replay_buffer, self.args.batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}, timesteps: {self.env.timesteps}")
                print(f"\033[0;31;43mTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}, timesteps: {self.env.timesteps}\033[0m")
                # Reset environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if episode_num > 4000:
                    break

                if episode_num%20==0:
                    replay_buffer.save(f"./results/buffer_{self.file_name}")


                if (t + 1) % self.args.eval_freq == 0:
                    if self.args.save_model: self.policy.save(f"./models/{self.file_name}")



class PegInHoleRL:
    def __init__(self, up_ctrl=None, parent=None, dt=0.1):
        self.up_ctrl = up_ctrl
        self.parent = parent
        self.dt = dt
        self.assemble_stage_flage = 0


        self.timesteps = 0

        self.max_episodes = 100

        action_high = np.array([0.0005, 0.0005])
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        # hole is [0.478, 0.3969]
        # +/- 0.02
        state_high = np.array([0.6, 0.55])
        state_low = np.array([0.3, 0.25])

        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)

        self.xy_tgt = np.array( [0.478, 0.3969] )

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
            # self.timer_stage12.timeout.connect(self.run_stage12)


        self.time_rl = 0

        self.D_rl = 0.02
        self.d_reach = 0.001
        self.episode_step = 0


    def set_label_ctrl_state(self, label_ctrl_state):
        self.label_ctrl_state = label_ctrl_state


    def run_stage12(self):

        # print('run_stage12 run')

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
        time.sleep(2)

        # random state
        self.parent.on_random_position()
        # r_min, r_max = 0.007, 0.015
        # r = r_max
        # angle =1/4 * np.pi
        # x = r * np.cos(angle) + self.xy_tgt[0]
        # y = r * np.sin(angle) + self.xy_tgt[1]
        #
        # random_pos = np.array([x, y, 0.18, 0, 0, 0]).astype(np.float64)
        # self.up_ctrl.connect_widget.apply_Rot(random_pos)
        time.sleep(2)

        self.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()

        # print('run_stage12 run')

        self.assemble_stage_flage = 0

        while self.assemble_stage_flage!=2:
            # print('self.assemble_stage_flage: ', self.assemble_stage_flage)

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
                    self.episode_step = 0

            if self.x_r[2]<0.1725 and error[2]>4.5:
                return 0

            self.x_r += dx
            self.up_ctrl.connect_widget.apply_Rot(self.x_r)

            time.sleep(0.1)

        # print('reset down!')

        self.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()

        self.d0 = np.linalg.norm(self.x_r[:2] - self.xy_tgt)

        # the state include the xy and force sensor
        state = self.x_r[:2]

        return state


    def step_pid(self, action):

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

            self.d = np.linalg.norm(self.x_r[:2] - self.xy_tgt)
            # keep the force in z axis constant
            # dx[2] = 5e-6 * np.sign(error[2])
            if np.abs(error[2]) > 0.5:
                dx[2] = 5e-6 * np.sign(error[2])

            if self.d < self.d_reach and force_contact_norm < 2:
                self.assemble_stage_flage = 3
                # self.label_ctrl_state.setText(f"Control: stage 3")

        self.error_last = error

        self.x_r += dx
        #
        self.up_ctrl.connect_widget.apply_Rot(self.x_r)

        # print('step')
        time.sleep(0.1)

        state = self.x_r[:2]
        done = self.assemble_stage_flage == 3
        action = dx[:2]
        if done:
            self.assemble_stage_flage = 0
            reward=2
        else:
            reward=0

        return state, reward, done, action

    def reaction_foce(self, pos):
        if pos[2]>0.55:
            return np.zeros(6)
        else:
            dx = 0.55-np.round(pos[2], 3)
            # print(dx)
            return np.array([0, 0, np.exp(100*dx)+3, 0, 0, 0])


    def env_pre(self):
        pass


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_rl(self, action):
        self.timesteps += 1
        force_contact_world = self.up_ctrl.ft.force_contact_world.copy()
        force_contact_norm = np.linalg.norm(force_contact_world[:3])
        torque_norm = np.linalg.norm(force_contact_world[3:])

        # if force_contact_norm > 40 or torque_norm > 0.5:
        #     return

        # only reaction to the force in z positive
        if force_contact_world[2] < 1:
            force_contact_world[2] = 0

        error = force_contact_world - self.F_r
        self.error_sum += error

        dx = np.zeros(6)

        if self.assemble_stage_flage == 2:  # to the hole
            dx = np.zeros(6)

            dx[:2] = action

            # keep the force in z axis constant
            # dx[2] = 5e-6 * np.sign(error[2])
            if np.abs(error[2]) > 0.5:
                dx[2] = 5e-6 * np.sign(error[2])

            if self.x_r[2] < 0.172 and force_contact_norm < 2:
                self.assemble_stage_flage = 3
                # self.label_ctrl_state.setText(f"Control: stage 3")

        self.error_last = error

        self.x_r += dx
        #
        self.up_ctrl.connect_widget.apply_Rot(self.x_r)

        # print('step')
        self.episode_step += 1
        time.sleep(0.1)


        self.x_r = self.up_ctrl.connect_widget.get_tgt_xyz_rot().copy()

        self.d = np.linalg.norm(self.x_r[:2] - self.xy_tgt)

        state = self.x_r[:2]
        reach_max_eposide = (self.episode_step >= self.max_episodes)
        reach_target = self.d < self.d_reach
        exceed_safe_reigion = self.d > self.D_rl

        done = reach_target or reach_max_eposide or exceed_safe_reigion
        if done:
            self.assemble_stage_flage = 0
            self.error_sum = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)
            if reach_target:
                # print('===== reach the target =====', self.d)
                print("\033[0;31;43m===== reach the target =====\033[0m", self.d)
                # reward = 2 - self.episode_step / self.max_episodes
                reward = 5 - (self.episode_step / self.max_episodes)*2
            else:
                print(self.d, self.d0)
                reward = max(1 - ((self.d - self.d0) / (self.D_rl - self.d0)), 0)

        else:
            reward = 0

        return state, reward, done

        pass
