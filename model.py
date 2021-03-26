import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Bernoulli, Categorical
from gym import spaces
from ltl2tree import LTL_OPS, OP2NARG
from utils import make_filter_image
import worlds.craft_world as craft


class BasePolicy(nn.Module):
    def __init__(self, input_size, output_state_size, config,
                 n_args=0, rnn_size=64, rnn_depth=1, has_arg=False):
        super(BasePolicy, self).__init__()

        self.n_args = n_args
        self.config = config
        self.has_arg = has_arg
        
        self.state_size = output_state_size
        self.rnn_size = rnn_size
        self.rnn_depth = rnn_depth

        # a linear layer that combines states from children
        if n_args:
            self.combine_state = nn.Linear(n_args*output_state_size, output_state_size)
            n_states = 2
        else:
            n_states = 1

        if config.env_name == 'Craft' and not config.use_gui:
            self.obs_linear = nn.Linear(input_size, rnn_size)
            rnn_input_size = rnn_size + output_state_size*n_states
        else:
            rnn_input_size = input_size + output_state_size*n_states

        # rnn for the symbol or operator
        if has_arg:
            self.combine_obs = nn.Linear(rnn_size+5, rnn_size)
        self.rnn = nn.GRU(rnn_input_size, rnn_size, self.rnn_depth)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # a linear layer that convert hidden states to interpretable vectors
        self.out_linear = nn.Linear(rnn_size*rnn_depth, output_state_size)

    def forward(self, inputs, args_obs, child_states, parent_state, hidden_state, masks,
                no_hidden=False):
        batch_size = inputs.shape[0]
        # prepare rnn inputs
        if len(child_states) > 0:
            child_states = torch.cat(child_states, dim=1)
            in_state = self.combine_state(child_states)
            in_state = in_state.to(self.config.device)
        if parent_state is None:
            parent_state = torch.zeros(batch_size, self.state_size)
        if hidden_state is None:
            hidden_state = torch.zeros(self.rnn_depth, batch_size, self.rnn_size)
        parent_state = parent_state.to(self.config.device)
        hidden_state = hidden_state.to(self.config.device)
        if self.config.env_name == 'Craft' and not self.config.use_gui:
            inputs = torch.relu(self.obs_linear(inputs))
        if args_obs is not None:
            inputs = torch.relu(self.combine_obs(torch.cat([inputs, args_obs], dim=1)))
        if len(child_states) > 0:
            rnn_in = torch.cat([inputs, in_state, parent_state], dim=1)
        else:
            rnn_in = torch.cat([inputs, parent_state], dim=1)
        # forward one rnn step
        rnn_in = rnn_in.unsqueeze(0)
        rnn_out, hidden_state = self.rnn(rnn_in * masks.view(1, -1, 1),
                                         hidden_state.detach() * masks.view(1, -1, 1))
        if no_hidden:
            hidden_state = torch.zeros(hidden_state.shape)
        # convert the hidden state to an interpretable vector
        flatten_hidden = hidden_state.permute(1,0,2).contiguous().view(batch_size, -1)
        out_state = self.out_linear(flatten_hidden)
        return rnn_out, hidden_state, out_state


class LangEmbedding(nn.Module):
    def __init__(self, symbol_size, emb_size=32, rnn_depth=1):
        super(LangEmbedding, self).__init__()
        input_size = symbol_size + 9  # including ops and parentheses
        self.input_size = input_size
        self.rnn_depth = rnn_depth
        self.rnn_size = emb_size
        self.rnn = nn.GRU(input_size, emb_size, rnn_depth)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.out_linear = nn.Linear(emb_size, emb_size)

    def forward(self, inputs):
        batch_size = 1
        hidden_state = torch.zeros(self.rnn_depth, batch_size, self.rnn_size)
        hidden_state = hidden_state.to(inputs.device)
        rnn_in = inputs.unsqueeze(1)
        rnn_out, hidden_state = self.rnn(rnn_in, hidden_state)
        rnn_out = rnn_out[-1]  # choose output from the last token
        rnn_out = self.out_linear(rnn_out)
        return rnn_out


class ImageEmbedding(nn.Module):
    def __init__(self, input_shape, output_dim=64, hidden_dim=64):
        super(ImageEmbedding, self).__init__()
        self._output_dim = output_dim
        self._conv1 = nn.Conv2d(3, hidden_dim, 3)
        self._conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3)
        self._conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3)
        self.cnn_output_shape = self._forward_cnn(torch.zeros(input_shape).unsqueeze(0)).shape
        cnn_output_dim = self._forward_cnn(torch.zeros(input_shape).unsqueeze(0)).nelement()
        self._lin = nn.Linear(cnn_output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self._conv2(x))
        x = torch.relu(self._conv3(x))
        x = nn.MaxPool2d(2)(x)
        return x

    def forward(self, x):
        x = self._forward_cnn(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = torch.relu(self._lin(x))
        return x


class LTLPolicy(nn.Module):
    def __init__(self, ltl_tree, symbols, args):
        super(LTLPolicy, self).__init__()

        self.ltl_tree = ltl_tree
        self.args = args

        if isinstance(args.observation_space, spaces.Tuple):
            if len(args.observation_space) == 3:
                # initialize for combined image and state value observation space
                input_size = args.image_emb_size + args.observation_space[1].shape[0]
            else:
                input_size = args.observation_space[0].shape[0]
            # get the cookbook to look up index
            self.cookbook = craft.Cookbook(args.recipe_path)
        else:
            input_size = args.observation_space.shape[0]
        if args.lang_emb:
            input_size += args.lang_emb_size

        # symbols and operators as nn modules
        self._modules = {}
        for symbol in symbols:
            if 'C_' in symbol:  # skip closer predicate
                continue
            self._modules[symbol] = BasePolicy(input_size,
                                               args.output_state_size,
                                               args,
                                               n_args=0,
                                               rnn_size=args.rnn_size,
                                               rnn_depth=args.rnn_depth)
            self.add_module(symbol, self._modules[symbol])
        if args.env_name == 'Craft':
            symbol = 'C'
            self._modules[symbol] = BasePolicy(input_size,
                                               args.output_state_size,
                                               args,
                                               n_args=0,
                                               rnn_size=args.rnn_size,
                                               rnn_depth=args.rnn_depth,
                                               has_arg=True)
            self.add_module(symbol, self._modules[symbol])
        if not args.baseline:
            for op in LTL_OPS:
                self._modules[op] = BasePolicy(input_size,
                                               args.output_state_size,
                                               args,
                                               n_args=OP2NARG[op],
                                               rnn_size=args.rnn_size,
                                               rnn_depth=args.rnn_depth)
                self.add_module(op, self._modules[op])
        # language embedding to encode ltl formulas
        if args.lang_emb:
            self.lang_emb = LangEmbedding(len(args.alphabets), emb_size=args.lang_emb_size)
        self.image_emb = None
        if isinstance(args.observation_space, spaces.Tuple):
            if len(args.observation_space) == 3:
                img_shape = args.observation_space[0].shape
                self.image_emb = ImageEmbedding((img_shape[2], img_shape[0], img_shape[1]),
                                                output_dim=args.image_emb_size)
        self.reset()

    def update_formula(self, ltl_tree, ltl_onehot=None):
        '''Set a new ltl_tree and update the fomula tree'''
        self.ltl_tree = ltl_tree
        self.ltl_onehot = ltl_onehot
        if self.ltl_onehot is not None:
            self.ltl_onehot = self.ltl_onehot.to(self.args.device)
        self.reset()

    def reset(self):
        '''Reset the module states'''
        self.prev_hidden_states = [None for _ in range(self.ltl_tree.size)]
        self.prev_parent_states = [None for _ in range(self.ltl_tree.size)]
        self.hidden_states = [None for _ in range(self.ltl_tree.size)]
        self.parent_states = [None for _ in range(self.ltl_tree.size)]

    def log_param(self, writer, iter):
        for key in self._modules.keys():
            print_key  = key.replace('!', 'not').replace('&', 'and').replace('|', 'or')
            for tag, value in self._modules[key].named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(print_key + '_' + tag,
                                     value.cpu().data.numpy(), iter)
                if value.grad is not None:
                    writer.add_histogram(print_key + '_' + tag+'/grad',
                                         value.grad.cpu().data.numpy(), iter)
        if self.image_emb:
            writer.add_image('image/conv1', make_filter_image(self.image_emb._conv1), iter)
            writer.add_image('image/conv2', make_filter_image(self.image_emb._conv2, use_color=False), iter)
            writer.add_image('image/conv3', make_filter_image(self.image_emb._conv3, use_color=False), iter)

    def forward_child(self, node, obs, args_obs, masks, no_hidden=False):
        values = node.value.split('_')
        value = values[0]
        if value in self._modules.keys():
            n_args = OP2NARG[value]
            child_states = []
            for i, child in enumerate(node.children):
                _, hidden_state, out_state = self.forward_child(child, obs, args_obs, masks, no_hidden)
                child_states.append(out_state)
            if len(values) == 1:
                arg = None
                in_args_obs = None
            else:
                arg = values[1]
                in_args_obs = args_obs[:,self.cookbook.get_index(arg)]
            rnn_out, hidden_state, out_state = \
                self._modules[value].forward(obs, in_args_obs, child_states,
                                             self.prev_parent_states[node.id],
                                             self.prev_hidden_states[node.id],
                                             masks,
                                             no_hidden)
            self.hidden_states[node.id] = hidden_state
            for child in node.children:
                self.parent_states[child.id] = out_state
            return rnn_out, hidden_state, out_state
        else:
            raise NotImplementedError
    
    def forward(self, obs, masks, no_hidden=False):
        # make image embedding if observation has images
        args_obs = None
        if type(obs) is tuple:
            if len(self.args.observation_space) == 3:
                img_obs = ((obs[0] / 255) - 0.5 / 0.5)
                if len(obs[0].shape) == 3:
                    img_obs = img_obs.unsqueeze(0)  # make the batch size
                img_emb = self.image_emb(img_obs.permute(0,3,1,2))
                if len(obs[1].shape) == 1:
                    pos_obs = obs[1].unsqueeze(0)
                else:
                    pos_obs = obs[1]
                if len(obs[2].shape) == 2:
                    args_obs = obs[2].unsqueeze(0)
                else:
                    args_obs = obs[2]
                obs = torch.cat((img_emb, pos_obs), 1)
            else:
                if len(obs[0].shape) == 1:
                    pos_obs = obs[0].unsqueeze(0)
                else:
                    pos_obs = obs[0]
                if len(obs[1].shape) == 2:
                    args_obs = obs[1].unsqueeze(0)
                else:
                    args_obs = obs[1]
                obs = pos_obs
        else:
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
        # make language embedding if needed
        if self.args.lang_emb:
            lang_out = self.lang_emb(self.ltl_onehot)
            lang_out = lang_out.repeat(obs.shape[0],1)
            obs = torch.cat((obs, lang_out), 1)
        rnn_out, _, _ = self.forward_child(self.ltl_tree, obs, args_obs, masks, no_hidden)
        self.prev_hidden_states = self.hidden_states
        self.prev_parent_states = self.parent_states
        return rnn_out.squeeze(0)


class LTLActorCritic(torch.nn.Module):
    def __init__(self, ltl_tree, symbols, args):
        super(LTLActorCritic, self).__init__()
        # base policy
        if args.baseline:
            symbols = ['all']
        self.base = LTLPolicy(ltl_tree, symbols, args)
        # actor: the final linear layer for action prediction
        if args.action_space.__class__.__name__ == "MultiBinary":
            num_outputs = args.action_space.shape[0]
            self.actor = Bernoulli(args.rnn_size, num_outputs)
        elif args.action_space.__class__.__name__ == "Discrete":
            num_outputs = args.action_space.n
            self.actor = Categorical(args.rnn_size, num_outputs)
        else:
            raise NotImplementedError
        # critic: the final linear layer to estimate the value function
        self.critic_linear = nn.Linear(args.rnn_size, 1)

    def update_formula(self, ltl_tree, ltl_onehot=None):
        '''Update the ltl_tree for both actor and critic'''
        self.base.update_formula(ltl_tree, ltl_onehot)

    def reset(self):
        self.base.reset()

    def log_param(self, writer, iter):
        self.base.log_param(writer, iter)
        for tag, value in self.critic_linear.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('critic_' + tag,
                                 value.cpu().data.numpy(), iter)
            if value.grad is not None:
                writer.add_histogram('critic_' + tag+'/grad',
                                     value.grad.cpu().data.numpy(), iter)

    def freeze(self, symbols):
        for name, param in self.base.named_parameters():
            name = name.split('.')[0]
            if name in symbols:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze(self):
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, obs):
        x = self.base(obs)
        return self.critic_linear(x), self.actor(obs), x

    def act(self, obs, masks, deterministic=False, no_hidden=False):
        x = self.base(obs, masks, no_hidden=no_hidden)
        value = self.critic_linear(x)
        dist = self.actor(x)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        x = self.base(inputs, masks)
        value = self.critic_linear(x)
        return value

    def evaluate_actions(self, inputs, masks, action):
        x = self.base(inputs, masks)
        dist = self.actor(x)
        value = self.critic_linear(x)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
