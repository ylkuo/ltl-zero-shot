import gym
import numpy as np
import random

from gym import spaces
from gym.utils import seeding
from spot2ba import Automaton
from ltl2tree import ltl2tree, ltl_tree_with_annotation_str, LTL_OPS
from ltl_sampler import ltl_sampler


def transition_to_onehot(trans, alphabets):
    seq = []
    alphabet2id = dict()
    idx = 0
    for alphabet in alphabets:
        alphabet2id[alphabet] = idx
        idx += 1
    for tran in trans:
        v = np.zeros(len(alphabets))
        for s in tran:
            v[alphabet2id[s]] = 1
        seq.append(v)
    return np.array(seq)


def onehot_to_transition(onehots, alphabets):
    seq = []
    id2alphabet = dict()
    idx = 0
    for alphabet in alphabets:
        id2alphabet[idx] = alphabet
        idx += 1
    for onehot in onehots:
        v = set()
        for idx, val in enumerate(onehot):
            if val > 0:
                v.add(id2alphabet[idx])
        seq.append(v)
    return seq


class CharStreamEnv(gym.Env):
    def __init__(self, formula, alphabets,
                 prefix_reward_decay=1., time_limit=10,
                 update_failed_trans_only=False):
        self.action_space = spaces.MultiBinary(len(alphabets))
        self.observation_space = spaces.MultiBinary(len(alphabets))
        self.prefix_reward_decay = prefix_reward_decay
        self.time_limit = time_limit
        self.seed()
        self.update_failed_trans_only = update_failed_trans_only
        # convert the ltl formula to a Buchi automaton
        if self.update_failed_trans_only:
            ltl_tree = ltl2tree(formula, alphabets)
            self._anno_formula, _, self._anno_maps = ltl_tree_with_annotation_str(ltl_tree, idx=0)
            self.ba = Automaton(self._anno_formula, alphabets)
        else:
            self.ba = Automaton(formula, alphabets)
        self._formula = formula
        self._alphabets = alphabets
        self._seq = []
        self._last_states = set(self.ba.get_initial_state())
        self._state_visit_count = 0
        # start the first game
        self.should_skip = False
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_data(self):
        return None

    def load(self):
        pass

    def step(self, action, random=False):
        if random:
            action = np.random.choice([0., 1.], len(self._alphabets))
        assert self.action_space.contains(action)
        self._seq.append(action)
        done = len(self._seq) >= self.time_limit  # done when reaching time limit
        trans = onehot_to_transition(self._seq, self._alphabets)
        is_prefix, dist_to_accept, last_states, failed_trans = \
                self.ba.is_prefix([trans[-1]], self._last_states)
        is_accpet = is_prefix and dist_to_accept < 0.1
        if is_accpet:  # not done even if it is in accept state
            reward = 1
            self._last_states = set([s for s in last_states if self.ba.is_accept(s)])
        elif is_prefix:
            if len(last_states.intersection(self._last_states)) > 0:
                self._state_visit_count += 1
            else:
                self._state_visit_count = 1
            self._last_states = last_states
            if self._state_visit_count == 1:
                reward = 0.1
            else:
                reward = 0.1 * (self.prefix_reward_decay ** (self._state_visit_count - 1))
        else:
            reward = -1
            done = True  # stay at done if the env doesn't reset
        if done and reward < 0.2:  # penalize if reaching time limit but not accept
            reward = -1
        info = {}
        # get the failed components if updating failed transitions only
        if self.update_failed_trans_only:
            components = set()
            for trans in failed_trans:
                for symbol in trans.split(' '):
                    symbol = symbol.replace('(', '').replace(')', '').replace('!', '')
                    if symbol in LTL_OPS: continue
                    if 'a_' in symbol:
                        components = components.union(self._anno_maps[symbol])
            info = {'failed_components': components}
        return np.array(action, dtype=np.int8), reward, done, info

    def reset(self):
        self._seq = []
        self._state_visit_count = 0
        self._last_states = set(self.ba.get_initial_state())
        return np.array([0 for _ in self._alphabets], dtype=np.int8)


if __name__ == '__main__':
    # sample a ltl formula
    alphabets = ['a', 'b', 'c']
    ltls = ltl_sampler(alphabets, n_samples=1)
    ltl, ba, _ = ltls[0]
    print('LTL formula:', ltl)
    ba.draw('tmp_images/ba.svg', show=False)
    states, trans = ba.gen_sequence()
    print('min sequence length:', ba.len_min_accepting_run)
    print('avg sequence length:', ba.len_avg_accepting_run)
    print('alphabets', ba._alphabets)
    print('states', states)
    print('trans', trans)
    one_hots = transition_to_onehot(trans, alphabets=ba._alphabets)
    print('one-hot:\n', one_hots)
    # run char stream environment
    env = CharStreamEnv(ltl, ba._alphabets)
    for one_hot in one_hots:
        print(one_hot, env.step(one_hot))
