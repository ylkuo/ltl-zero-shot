import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo.kfac import KFACOptimizer
from model import LTLActorCritic
from torch.optim.lr_scheduler import MultiStepLR


class A2CTrainer(object):
    def __init__(self, ltl_tree, symbols, args):
        actor_critic = LTLActorCritic(ltl_tree, symbols, args)
        actor_critic.to(args.device)

        self.actor_critic = actor_critic
        self.args = args

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        if args.use_lr_scheduler:
            self.scheduler = MultiStepLR(self.optimizer,
                milestones=[args.lr_scheduled_update*args.log_interval], gamma=0.1)

    def update_formula(self, ltl_tree, ltl_onehot=None):
        self.actor_critic.update_formula(ltl_tree, ltl_onehot)

    def update(self, rollouts):
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values = []
        action_log_probs = []
        dist_entropy = []
        self.actor_critic.reset()  # reset to set the starting hidden states correct
        for i in range(num_steps):
            i_values, i_action_log_probs, i_dist_entropy = self.actor_critic.evaluate_actions(
                rollouts.get_obs(i),
                rollouts.masks[i],
                rollouts.actions[i])
            values.append(i_values)
            action_log_probs.append(i_action_log_probs)
            dist_entropy.append(i_dist_entropy)

        values = torch.stack(values)
        action_log_probs = torch.stack(action_log_probs)
        dist_entropy = torch.stack(dist_entropy).mean()

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()
        if self.args.use_lr_scheduler:
            self.scheduler.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
