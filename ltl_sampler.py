import numpy as np
import random
import worlds.craft_world as craft

from easydict import EasyDict as edict
from itertools import combinations
from ltl2tree import *
from spot2ba import Automaton


def add_basic_ltl(alphabets):
    ltls = []
    for a in alphabets:
        ba = Automaton(a, alphabets)
        ltls.append((a, ba, None))
    for key, val in OP2NARG.items():
        if key == '!': continue
        if val == 1:
            ltl = key + ' ' + np.random.choice(alphabets)
        else:
            args = np.random.choice(alphabets, 2, replace=False)
            ltl = args[0] + ' ' + key + ' ' + args[1]
        ba = Automaton(ltl, alphabets)
        ltls.append((ltl, ba, None))
    return ltls


def permute(alphabets, ltl):
    tokens = ltl.split(' ')
    for i, token in enumerate(tokens):
        flip = np.random.choice([True, False], p=[0.3, 0.7])
        if flip:
            if token in alphabets:
                tokens[i] = np.random.choice(alphabets)
            elif token in OP_1:
                tokens[i] = np.random.choice(OP_1)
            elif token in OP_2:
                tokens[i] = np.random.choice(OP_2)
    # sample to add a `not'
    out_tokens = []
    for i, token in enumerate(tokens):
        if token in alphabets or token == '(':
            add_not = np.random.choice([True, False], p=[0.05, 0.95])
            if add_not:
                out_tokens.append('!')
        out_tokens.append(token)
    return ' '.join(out_tokens)


def get_new_ltl(alphabets, ltl, ba, n_steps, n_accept, env_name=''):
    n_samples = 100
    # sample n_samples accepting sequence from current ltl
    seq = []
    for i in range(n_samples):
        states, trans = ba.gen_sequence()
        seq.append(trans)
    accepting_ratio = 1
    new_ltl = None; new_ba = None
    count = 0
    while accepting_ratio > 0.1 and count < 15:
        # permute the formula
        new_ltl = permute(alphabets, ltl)
        new_ba = Automaton(new_ltl, alphabets)
        if new_ba.n_states <= 1 or \
                new_ba.num_accept_str(n_steps) > n_accept or \
                check_all_same(alphabets, new_ltl, new_ba, env_name):
            continue
        # check if the accpting ratio is low enough
        n_acc = 0
        for trans in seq:
            if new_ba.recognize(trans):
                n_acc += 1
        accepting_ratio = n_acc / len(seq)
        count += 1
    if count >= 15:
        return None, None
    return new_ltl, new_ba


def check_all_same(alphabets, ltl, ba, env_name=''):
    n_trans = 15
    if env_name == 'Craft':
        # Only check for the all-empty case
        tokens = set()
        trans = [tokens]
        for i in range(n_trans):
            if ba.recognize(trans):
                return True
            trans.append(tokens)
        return False
    else:
        # Check all possible combinations
        for check_len in range(len(alphabets)+1):
            for tokens in combinations(alphabets, check_len):
                tokens = list(tokens)
                trans = [tokens]
                for i in range(n_trans):
                    if ba.recognize(trans):
                        return True
                    trans.append(tokens)
        return False


def check_should_add(ltl, ba, include_templates, skip_templates, env_name=''):
    # add the ltl in templates
    if len(include_templates) > 0:
        if ltlstr2template(ltl) in include_templates:
            return True
    # reject if 1) no accepting state, 2) too few states
    #           3) too simple, i.e. one step to acceptance
    #           4) the templates we want to skip
    if ba.has_accept and \
            not np.isinf(ba.len_min_accepting_run) and \
            ba.n_states > 1 and \
            ltlstr2template(ltl) not in skip_templates:
        if env_name == 'Craft':
            args = edict({
                    'recipe_path': 'worlds/craft_recipes_basic.yaml',
                    'formula': ltl,
                    'prefix_reward_decay': 0.8,
                    'num_steps': 15,
                    'target_fps': None,
                    'use_gui': True,
                    'is_headless': True
                })
            env, _ = craft.sample_craft_env(args, n_retries=5, max_n_seq=100, goal_only=False)
            if env is None:
                print(' Bad env')
            return env is not None
        return True
    return False


def ltl_sampler(alphabets, env_name='',
                n_samples=1,
                skip_templates=[],
                include_templates=[],
                add_basics=False,
                min_symbol_len=1,
                max_symbol_len=10,
                n_steps=15,
                n_accept=10**9,
                paired_gen=False):
    filtered_alphabets = [a for a in alphabets if 'C_' not in a]
    if len(filtered_alphabets) == 5:
        n_accept = 10**21  # for 5 symbols
    elif len(filtered_alphabets) == 7:
        n_accept = 3 * 10**29  # for craft, 7 symbols
    elif len(filtered_alphabets) == 9:
        n_accept = 10**38  # for 9 symbols
    cfg = get_ltl_grammar(alphabets, env_name)
    ltls = []; considered = set()
    if add_basics:
        ltls = add_basic_ltl(alphabets)
        considered = set([ltl for ltl, _, _ in ltls])
    if paired_gen:
        n_samples = int((n_samples-len(ltls)) / 2)
    else:
        n_samples = n_samples - len(ltls)
    n_acc = 0; n_rej = 0
    for i in range(n_samples):
        print('Generate {}th formula'.format(i))
        while True:
            # generate LTL formula (including its pair)
            ltl = generate_ltl(cfg, env_name=env_name)
            symbols = [s for s in ltl.split(' ') if s != ')' and s != '(']
            sym_alphabets = [s for s in symbols if s in alphabets]
            # restrict craft max symbols to be 6 since spot may slow down at 12
            # and we have C_p & p for the craft env
            if len(symbols) <= min_symbol_len or len(symbols) > max_symbol_len:
                n_rej += 1
                considered.add(ltl)
                continue
            ba = Automaton(ltl, filtered_alphabets)
            #print(' consider', ltl, ba.num_accept_str(n_steps))
            if ba.n_states <= 1 or \
                    ltl in considered or \
                    np.isinf(ba.len_min_accepting_run) or \
                    ba.num_accept_str(n_steps) > n_accept or \
                    check_all_same(alphabets, ltl, ba, env_name) or \
                    ba.len_avg_accepting_run < 1:
                n_rej += 1
                considered.add(ltl)
                #print(' reject', ltl, ba.num_accept_str(n_steps))
                continue
            new_ltl = None
            if paired_gen:
                new_ltl, new_ba = get_new_ltl(alphabets, ltl, ba, n_steps, n_accept, env_name)
                if new_ltl is None or new_ltl in considered:
                    print(' reject', new_ltl)
                    n_rej += 1
                    continue
            # add the generated formulas
            if check_should_add(ltl, ba, include_templates, skip_templates, env_name=env_name):
                if paired_gen:
                    if check_should_add(new_ltl, new_ba, include_templates, skip_templates, env_name=env_name):
                        print(' add {}: {}'.format(new_ltl, new_ba.num_accept_str(n_steps)))
                        considered.add(new_ltl)
                        new_ltl = replace_symbols(new_ltl, env_name=env_name)
                        new_ba = Automaton(new_ltl, alphabets)
                        if new_ba.len_avg_accepting_run < 1:
                            continue
                        ltls.append((new_ltl, new_ba, ltl))
                    else:
                        print(' reject', new_ltl)
                        n_rej += 1
                        continue
                print(' add {}: {}'.format(ltl, ba.num_accept_str(n_steps)))
                n_acc += 1
                considered.add(ltl)
                ltl = replace_symbols(ltl, env_name=env_name)
                ba = Automaton(ltl, alphabets)
                if ba.len_avg_accepting_run < 1:
                    continue
                ltls.append((ltl, ba, new_ltl))
                break
            else:
                n_rej += 1
                print(' reject', ltl)
    print('Acceptance rate: {}'.format(n_acc / (n_acc+n_rej)))
    ltls.sort(key=lambda x: (len(x[0]), x[1].len_avg_accepting_run))
    return ltls


if __name__ == '__main__':
    ltl_formula = '( G F ( ( C_workbench U workbench ) & ( C_gem U gem ) ) )'
    # parse ltl formula to cfg tree and then convert to an expression tree
    alphabets = ['C_workbench', 'workbench', 'C_gem', 'gem']
    cfg_tree = parse_ltl(ltl_formula, alphabets)
    print(cfg_tree)
    ltl_tree, idx = convert_ltl_tree(cfg_tree)
    print(ltl_tree_str(ltl_tree))
    args = edict({
        'recipe_path': 'worlds/craft_recipes_basic.yaml',
        'formula': ltl_formula,
        'prefix_reward_decay': 0.8,
        'num_steps': 15,
        'target_fps': None,
        'use_gui': True,
        'is_headless': True
    })
    env = craft.sample_craft_env(args, n_retries=5)
    if env is None:
        print('Bad env')
