import gym
import os
import torch

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from worlds.craft_world import sample_craft_env
from worlds.char_stream import CharStreamEnv


def make_single_env(args, env_data, n_retries=5, max_n_seq=50):
    if args.env_name == 'CharStream':
        env = CharStreamEnv(args.formula, args.alphabets,
                            prefix_reward_decay=args.prefix_reward_decay,
                            time_limit=args.num_steps,
                            update_failed_trans_only=args.update_failed_trans_only)
    elif args.env_name == 'Craft':
        env, _ = sample_craft_env(args, env_data=env_data, train=True, n_steps=3, \
                n_retries=n_retries, max_n_seq=max_n_seq, goal_only=True)
        if env_data is not None:
            env.load(env_data)
    return env


def make_env(env_id, args, seed, rank, log_dir, allow_early_resets, env_data=None):
    def _thunk():
        env = make_single_env(args, env_data)
        env.seed(seed + rank)
        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)
        return env

    return _thunk


def make_vec_envs(args,
                  device,
                  allow_early_resets,
                  env_data=None):
    envs = [
        make_env(args.env_name, args, args.seed, i, args.log_dir, allow_early_resets, env_data)
        for i in range(args.num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        if type(obs) is dict:
            out_obs = []
            for i, s in obs.items():
                out_obs.append(torch.from_numpy(s).float().to(self.device))
        else:
            out_obs = torch.from_numpy(obs).float().to(self.device)
        return out_obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        else:
            actions = actions.squeeze(0)  # make the first dimension the same as number of workers
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if type(obs) is dict:
            out_obs = []
            for i, s in obs.items():
                out_obs.append(torch.from_numpy(s).float().to(self.device))
        else:
            out_obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return out_obs, reward, done, info
