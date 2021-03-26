import argparse
import copy
import glob
import imageio
import numpy as np
import os
import pickle
import random
import shutil
import time
import torch
import utils
import worlds.craft_world as craft

from a2c import A2CTrainer
from collections import deque
from datetime import datetime
from envs import make_vec_envs, make_single_env
from ltl2tree import ltl2tree, ltlstr2template, ltl2onehot, LTL_OPS
from ltl_sampler import ltl_sampler
from spot2ba import Automaton
from storage import RolloutStorage
from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='RL with LTL')
    parser.add_argument('--algo', default='a2c', help='algorithm to use: a2c')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False,
                        help='use learning rate scheduler or not (default: False)')
    parser.add_argument('--lr_scheduled_update', type=int, default=300,
                        help='number of grident updates before chaning learning rate (default: 300)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--env_name', default='CharStream',
                        help='environment to train on: CharStream | Craft')
    parser.add_argument('--num_train_ltls', type=int, default=50,
                        help='number of sampled ltl formula for training (default: 50)')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='number of forward steps in A2C (default: 10)')
    parser.add_argument('--num_env_steps', type=int, default=5*10e3,
                        help='number of environment steps to train per environment (default: 10e5)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to go over all formulas (default: 10)')
    parser.add_argument('--log_dir', default='/tmp/ltl-rl/',
                        help='directory to save agent logs (default: /tmp/ltl-rl)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='dimensions of the RNN hidden layers.')
    parser.add_argument('--rnn_depth', type=int, default=1,
                        help='number of layers in the stacked RNN.')
    parser.add_argument('--output_state_size', type=int, default=32,
                        help='dimensions of the output interpretable state vector.')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--prefix_reward_decay', type=float, default=0.03,
                        help='decay of reward if following prefix (default: 0.03)')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--save_dir', default='./models/',
                        help='directory to save agent logs (default: ./models/)')
    parser.add_argument('--save_model_name', default='model.pt',
                        help='name of the saved model (default: model.pt)')
    parser.add_argument('--load_model_name', default='model.pt',
                        help='name of the model to be loaded (default: model.pt)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='train and evaluate baseline model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='in training mode or not')
    parser.add_argument('--load_formula_pickle', action='store_true', default=False,
                        help='train and evaluate baseline model')
    parser.add_argument('--formula_pickle', default='',
                        help='path to load the formulas')
    parser.add_argument('--test_formula_pickle_1', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_2', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_3', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_4', default='',
                        help='path to load the test formulas')
    parser.add_argument('--save_env_data', action='store_true', default=False,
                        help='save environment data')
    parser.add_argument('--load_env_data', action='store_true', default=False,
                        help='load environment data')
    parser.add_argument('--env_data_path', default='./data/env.pickle',
                        help='path to load environment data (default: ./data/env.pickle)')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='load pretrained model')
    parser.add_argument('--lang_emb', action='store_true', default=False,
                        help='train the language embedding baseline')
    parser.add_argument('--lang_emb_size', type=int, default=32,
                        help='embedding size of the ltl formula (default: 32)')
    parser.add_argument('--image_emb_size', type=int, default=64,
                        help='embedding size of the input image (default: 64)')
    parser.add_argument('--min_epoch', type=int, default=0,
                        help='starting epoch to evaluate (default: 0)')
    parser.add_argument('--min_formula', type=int, default=0,
                        help='starting formula to evaluate (default: 0)')
    parser.add_argument('--gen_formula_only', action='store_true', default=False,
                        help='only generate the training/testing formulas')
    parser.add_argument('--load_eval_train', action='store_true', default=False,
                        help='load the models in the folder first, run eval, and then train from the last one')
    parser.add_argument('--summary_dir', default='runs/',
                        help='path to save tensorboard summary')
    # test
    parser.add_argument('--num_test_ltls', type=int, default=50,
                        help='number of sampled ltl formula for testing (default: 50)')
    parser.add_argument('--test_in_domain', action='store_true', default=False,
                        help='test formula that is in training templates')
    parser.add_argument('--test_out_domain', action='store_true', default=False,
                        help='test formula that is not in training templates')
    parser.add_argument('--max_symbol_len', type=int, default=10,
                        help='max number of nodes in formula (default: 10)')
    parser.add_argument('--min_symbol_len', type=int, default=1,
                        help='min number of nodes in formula (default: 1)')
    parser.add_argument('--no_time', action='store_true', default=False,
                        help='evaluate no time dependency')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def setup_summary_writer(args):
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M")
    lr_str = str(args.lr).split('.')[1]
    alpha_str = str(args.alpha).split('.')[1]
    entropy_str = str(args.entropy_coef).split('.')[1]
    dir_name = args.summary_dir + 'env=%s_seed=%s_datetime=%s_nformula=%i_lr=%s_alpha=%s_entropy=%s_rnnsize=%d_rnndepth=%d_algo=%s' \
        % (args.env_name, args.seed, now_str, args.num_train_ltls, lr_str, alpha_str, entropy_str, args.rnn_size, args.rnn_depth, args.algo)
    if args.lang_emb:
        dir_name = dir_name + '_langemb'
    elif args.no_time:
        dir_name = dir_name + '_notime'
    elif args.baseline:
        dir_name = dir_name + '_baseline'
    shutil.rmtree(dir_name, ignore_errors=True)
    args.writer = SummaryWriter(dir_name)
    return args


def save_formulas(formulas, file_path):
    formulas = [(f[0], f[2]) for f in formulas]
    with open(file_path, 'wb') as f:
        pickle.dump(formulas, f)


def load_formulas(file_path, alphabets):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    formulas = [(f[0], Automaton(f[0], alphabets), f[1]) for f in data]  # use None for Buchi
    return formulas


def get_num_files(args):
    folder = args.save_model_name.split('/')[0]
    save_path = os.path.join(args.save_dir, args.algo)
    folder_path = os.path.join(save_path, folder)
    print('Folder path', folder_path)
    files = [f for f in glob.glob(folder_path + '/*.pt') if 'best' not in f]
    return len(files)


def train(args, formulas):
    device = torch.device(utils.choose_gpu() if args.cuda else "cpu")
    args.device = device
    # init formula
    for formula, _, _ in formulas:
        args.formula = formula
        break
    ltl_tree = ltl2tree(args.formula, args.alphabets)
    # get init formula and set up the training
    envs = make_vec_envs(args, device, False)
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space

    if args.algo == 'a2c':
        agent = A2CTrainer(ltl_tree, args.alphabets, args)
    else:
        raise NotImplementedError

    if args.load_eval_train:
        n_models = get_num_files(args)
        if n_models > 0:
            args.load_model = True
            args.load_model_name = args.save_model_name + '_' + str(n_models-1) + '.pt'
    else:
        n_models = 0

    if args.load_model:
        model_path = os.path.join(args.save_dir, args.algo, args.load_model_name)
        print('Load model:', model_path)
        agent.actor_critic.load_state_dict(torch.load(model_path)[0])

    if args.load_env_data:
        with open(args.env_data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []
    test_formulas = []
    if args.test_formula_pickle_1 != '':
        test_formulas.append(load_formulas(args.test_formula_pickle_1, args.alphabets))
    if args.test_formula_pickle_2 != '':
        test_formulas.append(load_formulas(args.test_formula_pickle_2, args.alphabets))
    if args.test_formula_pickle_3 != '':
        test_formulas.append(load_formulas(args.test_formula_pickle_3, args.alphabets))
    if args.test_formula_pickle_4 != '':
        test_formulas.append(load_formulas(args.test_formula_pickle_4, args.alphabets))
    n_update = 0; n_iters = 0; best_accuracy = 0.
    example_formula = formulas[0][0]
    for e in range(args.num_epochs):
        if e < args.min_epoch: continue
        random.shuffle(formulas)
        for i, f in enumerate(formulas):
            if i < args.min_formula: continue
            args.min_formula = 0  # reset after passing the bar
            if n_update < n_models:  # test formulas first if has some pretrained models
                print('Evaluate update', n_update)
                for j, test_formula in enumerate(test_formulas):
                    args.formula = example_formula
                    n_successes, final_steps, n_formula = test(args, test_formula,
                           model_name=args.save_model_name + '_' + str(n_update) + '.pt')
                    args.writer.add_scalar('accuracy_'+str(j), float(n_successes)/n_formula, n_update)
                    args.writer.add_histogram('n_steps_'+str(j), final_steps, n_update)
                n_update += 1
                continue
            if args.no_time:
                exit()
            formula, ba, _ = f
            envs.close(); del envs
            args.formula = formula
            print('Process formula {}'.format(formula))
            if args.load_env_data and len(data) > 0:
                envs = make_vec_envs(args, args.device, True, data[i])
            else:
                envs = make_vec_envs(args, args.device, True)
            env = make_single_env(args, None)
            if env.should_skip:
                print('Skip bad env for {}'.format(formula))
                continue
            if args.save_env_data:
                writer = args.writer; args.writer = None
                tmp_args = copy.deepcopy(args)
                tmp_args.num_processes = 1
                tmp_envs = make_vec_envs(tmp_args, args.device, True)
                data.append(tmp_envs.venv.envs[0].get_data())
                tmp_envs.close(); del tmp_envs
                args.writer = writer
            ltl_tree = ltl2tree(args.formula, args.alphabets, args.baseline)
            if args.lang_emb:
                agent.update_formula(ltl_tree, ltl2onehot(args.formula, args.alphabets))
            else:
                agent.update_formula(ltl_tree)
            print("Train epoch {}, {}th formula {}, length {}"
                .format(e, i, args.formula, ba.len_avg_accepting_run))
            train_formula(args, agent, envs, n_epoch=e, n_formula=i)
            n_iters += 1
            if n_iters % args.log_interval == 0:
                agent.actor_critic.log_param(args.writer, n_update)
                if 'lr' in agent.optimizer.param_groups[0]:
                    args.writer.add_scalar('learning_rate',
                        agent.optimizer.param_groups[0]['lr'],
                        n_update)
                # save model
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
                model_path = os.path.join(save_path, args.save_model_name + '_' + str(n_update) + '.pt')
                torch.save([
                    agent.actor_critic.state_dict(),
                    agent.optimizer.state_dict()
                ], model_path)
                print('Save model {}'.format(model_path))
                # test formulas (in and out domain)
                for j, test_formula in enumerate(test_formulas):
                    args.formula = example_formula
                    n_successes, final_steps, n_formula = test(args, test_formula,
                           model_name=args.save_model_name + '_' + str(n_update) + '.pt')
                    accuracy = float(n_successes)/n_formula
                    args.writer.add_scalar('accuracy_'+str(j), accuracy, n_update)
                    args.writer.add_histogram('n_steps_'+str(j), final_steps, n_update)
                    if j == 1 and accuracy > best_accuracy:
                        model_path = os.path.join(save_path, args.save_model_name + '_best.pt')
                        torch.save([
                            agent.actor_critic.state_dict(),
                            agent.optimizer.state_dict(),
                            n_update,
                            accuracy
                        ], model_path)
                        best_accuracy = accuracy
                n_update += 1
    if args.save_env_data:
        with open(args.env_data_path, 'wb') as f:
            pickle.dump(data, f)
    envs.close(); del envs  # close the env so no EOF error


def train_formula(args, agent, envs, n_epoch=0, n_formula=0):
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space, envs.action_space)

    episode_rewards = deque(maxlen=10)

    count = 0
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        start = time.time()
        agent.actor_critic.reset()
        obs = envs.reset()
        if type(rollouts.obs) is list:
            for i in range(len(rollouts.obs)):
                rollouts.obs[i][0].copy_(obs[i])
        else:
            rollouts.obs[0].copy_(obs)
        rollouts.to(args.device)
        rollouts.step = 0
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = agent.actor_critic.act(
                    rollouts.get_obs(step), rollouts.masks[step], deterministic=False)

            # Observation, reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)
            # Reset hidden state if done
            if agent.actor_critic.base.hidden_states[0] is not None:
                for i, done_ in enumerate(done):
                    if done_:
                        for k, _ in enumerate(agent.actor_critic.base.hidden_states):
                            hidden_size = agent.actor_critic.base.hidden_states[k][args.rnn_depth-1][i].shape
                            agent.actor_critic.base.hidden_states[k][args.rnn_depth-1][i] = torch.zeros(hidden_size)

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.get_obs(-1), rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        if len(episode_rewards) > 1:
            total_num_steps = args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                " entropy {}, value loss {}, action loss {}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))


def sample_formulas_train(args):
    if args.env_name == 'CharStream':
        args.alphabets = ['a', 'b', 'c', 'd', 'e']
    elif args.env_name == 'Craft':
        args.recipe_path = 'worlds/craft_recipes_basic.yaml'
        args.alphabets = craft.get_alphabets(args.recipe_path)
        args.use_gui = False
        args.is_headless = True
        args.target_fps = None
    if os.path.isfile(args.formula_pickle) or args.load_formula_pickle:
         return load_formulas(args.formula_pickle, args.alphabets)
    if args.env_name == 'CharStream':
        formulas = ltl_sampler(args.alphabets,
                               env_name=args.env_name,
                               n_samples=args.num_train_ltls,
                               min_symbol_len=args.min_symbol_len,
                               max_symbol_len=args.max_symbol_len,
                               paired_gen=True,
                               n_steps=args.num_steps)
    elif args.env_name == 'Craft':
        formulas = ltl_sampler(args.alphabets,
                               env_name=args.env_name,
                               n_samples=int(args.num_train_ltls*1.3),
                               min_symbol_len=args.min_symbol_len,
                               max_symbol_len=args.max_symbol_len,
                               paired_gen=False,
                               add_basics=False,
                               n_steps=args.num_steps)
    # filter formulas
    if args.env_name == 'Craft':
        formulas = [(f, ba, paired_f) for f, ba, paired_f in formulas \
                if craft.check_excluding_formula(f, args.alphabets, args.recipe_path)]
        # random sample out of domain
        test_formulas = random.choices(formulas, k=args.num_test_ltls)
        save_formulas(test_formulas, args.test_formula_pickle_1)
        formulas = [f for f in formulas if f not in test_formulas]
        if len(formulas) > args.num_train_ltls:
            formulas = formulas[:args.num_train_ltls]
    save_formulas(formulas, args.formula_pickle)
    return formulas


def sample_formulas_test(args):
    if args.env_name == 'CharStream':
        args.alphabets = ['a', 'b', 'c', 'd', 'e']
    elif args.env_name == 'Craft':
        args.recipe_path = 'worlds/craft_recipes_basic.yaml'
        args.alphabets = craft.get_alphabets(args.recipe_path)
        args.is_headless = True
        args.use_gui = False
        args.target_fps = None
    if os.path.isfile(args.test_formula_pickle_1) or args.load_formula_pickle:
        return load_formulas(args.test_formula_pickle_1, args.alphabets)
    train_formulas = load_formulas(args.formula_pickle, args.alphabets)
    train_templates = set([ltlstr2template(f) for f, _, _ in train_formulas])
    if args.test_in_domain:
        formulas = random.sample(train_formulas, args.num_test_ltls)
        save_formulas(formulas, args.test_formula_pickle_1)
        return formulas
    elif args.test_out_domain:
        include_templates = []
        skip_templates = train_templates
    else:
        include_templates = []
        skip_templates = []
    if args.env_name == 'CharStream':
        formulas = ltl_sampler(args.alphabets,
                               env_name=args.env_name,
                               n_samples=args.num_test_ltls,
                               include_templates=include_templates,
                               skip_templates=skip_templates,
                               min_symbol_len=args.min_symbol_len,
                               max_symbol_len=args.max_symbol_len,
                               n_steps=args.num_steps)
    elif args.env_name == 'Craft':
        formulas = ltl_sampler(args.alphabets,
                               env_name=args.env_name,
                               n_samples=int(args.num_test_ltls*1.2),
                               include_templates=include_templates,
                               skip_templates=skip_templates,
                               min_symbol_len=args.min_symbol_len,
                               max_symbol_len=args.max_symbol_len,
                               n_steps=args.num_steps)
    # filter formulas
    if args.env_name == 'Craft':
        formulas = [(f, ba, paired_f) for f, ba, paired_f in formulas \
                if craft.check_excluding_formula(f, args.alphabets, args.recipe_path)]
    save_formulas(formulas, args.test_formula_pickle_1)
    return formulas


def test(args, formulas, model_name, save_gif=False):
    if save_gif:
        args.return_screen = True
    if not args.device:
        device = torch.device(choose_gpu() if args.cuda else "cpu")
        args.device = device
    # load test env
    data = []
    if args.load_env_data:
        with open(args.env_data_path, 'rb') as f:
            data = pickle.load(f)
    # test for each formula
    env = make_single_env(args, None, max_n_seq=100)
    # load model
    model_path = os.path.join(args.save_dir, args.algo, model_name)
    ltl_tree = ltl2tree(args.formula, args.alphabets, args.baseline)
    args.observation_space = env.observation_space
    args.action_space = env.action_space
    if args.algo == 'a2c':
        agent = A2CTrainer(ltl_tree, args.alphabets, args)
    else:
        raise NotImplementedError
    agent.actor_critic.load_state_dict(torch.load(model_path, map_location=args.device)[0])
    agent.actor_critic.eval()
    n_successes = 0; n_formula = 0
    final_steps = np.zeros(len(formulas))
    for i, i_formula in enumerate(formulas):
        formula, ba, _ = i_formula
        env.close(); del env
        args.formula = formula
        env = make_single_env(args, None, max_n_seq=100)
        if env.should_skip:
            final_steps[i] = 1
            if not save_gif:
                print('Skip {} because of bad env'.format(args.formula))
            continue
        if args.load_env_data and len(data) > 0:
            env.load(data[i])
        if args.save_env_data:
            data.append(env.get_data())
        ltl_tree = ltl2tree(args.formula, args.alphabets, args.baseline)
        if not save_gif:
            print("Test formula {}, {}".format(args.formula, env._formula))
        if args.lang_emb:
            agent.update_formula(ltl_tree, ltl2onehot(args.formula, args.alphabets))
        else:
            agent.update_formula(ltl_tree)
        screens = []
        with torch.no_grad():
            n_formula += 1
            agent.actor_critic.reset()
            obs = env.reset()
            done = False; accumulated_reward = 0
            actions = []
            for step in range(args.num_steps):
                # Sample actions
                if type(obs) is dict:
                    test_obs = []
                    for _, s in obs.items():
                        test_obs.append(torch.FloatTensor(s))
                        test_obs[-1] = test_obs[-1].to(args.device)
                    test_obs = tuple(test_obs)
                else:
                    test_obs = torch.FloatTensor(obs)
                    test_obs = test_obs.to(args.device)
                mask = torch.FloatTensor([1.0])
                mask = mask.to(args.device)
                _, action, _ = agent.actor_critic.act(test_obs, mask,
                    deterministic=True, no_hidden=args.no_time)
                # Observation, reward and next obs
                obs, reward, done, infos = env.step(action[0])
                actions.append(action[0])
                #print('  ', reward, done, action[0])
                accumulated_reward += reward
                if save_gif:
                    screens.append(infos['screen'])
                if done:
                    final_steps[i] = step
                    break
        if step == args.num_steps - 1 and accumulated_reward > 1.6:
            n_successes += 1
            if save_gif:
                imageio.mimsave('tmp_images/movie_'+ str(i) +'.gif', screens, fps=5)
                print('{}\t{}'.format(i, args.formula))
            else:
                print(' Success', accumulated_reward) #, actions)
    if not save_gif:
        print('Accuracy: {} ({}/{})'.format(n_successes/n_formula, n_successes, n_formula))
    if args.save_env_data:
        with open(args.env_data_path, 'wb') as f:
            pickle.dump(data, f)
    env.close(); del env  # close the env so no EOF error
    return n_successes, final_steps, n_formula


def main():
    args = get_args()

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)
    torch.set_num_threads(1)

    if args.gen_formula_only:
        if args.test_in_domain or args.test_out_domain:
            formulas = sample_formulas_test(args)
        else:
            formulas = sample_formulas_train(args)
        exit()

    args.return_screen = False
    if args.train:
        args = setup_summary_writer(args)
        formulas = sample_formulas_train(args)
        args.formula, _, _ = formulas[0]
        train(args, formulas)
        print('Finish training')
    else:
        device = torch.device(utils.choose_gpu() if args.cuda else "cpu")
        args.device = device
        formulas = sample_formulas_test(args)
        args.formula, _, _ = formulas[0]
        n_successes, _, _ = test(args, formulas, model_name=args.save_model_name, save_gif=True)
        print('Accuracy:', float(n_successes/len(formulas)))


if __name__ == '__main__':
    main()
