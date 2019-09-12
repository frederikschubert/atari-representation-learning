import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import wandb

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.model import NNBase, Policy
from a2c_ppo_acktr.storage import RolloutStorage
from aari.envs import make_vec_envs
from src.encoders import ImpalaCNN, NatureCNN
from src.utils import get_argparser


def get_envs(
    env_name, seed=42, num_processes=1, num_frame_stack=1, downsample=False, color=False
):
    return make_vec_envs(
        env_name, seed, num_processes, num_frame_stack, downsample, color
    )


def get_encoder(args, observation_shape, device):
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)

    if args.weights_path == "None":
        sys.stderr.write(
            "Training without loading in encoder weights! Are sure you want to do that??"
        )
    else:
        print(
            "Print loading in encoder weights from probe of type {} from the following path: {}".format(
                args.method, args.weights_path
            )
        )
        encoder.load_state_dict(torch.load(args.weights_path, map_location=device))
        encoder.eval()
    return encoder


class SimpleBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super().__init__(recurrent, num_inputs, hidden_size)
        init_ = lambda m: utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if recurrent:
            num_inputs = hidden_size

        # self.actor = init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        self.critic_linear = init_(nn.Linear(num_inputs, 1))
        self.train()
    
    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs

def get_agent(args, envs, encoder, device):
    actor_critic = Policy([encoder.feature_size], envs.action_space, base=SimpleBase)
    actor_critic.to(device)
    agent = algo.PPO(
        actor_critic,
        args.ppo_clip_param,
        args.ppo_epoch,
        args.ppo_num_mini_batch,
        args.ppo_value_loss_coef,
        args.ppo_entropy_coef,
        lr=args.ppo_lr,
        eps=args.ppo_eps,
        max_grad_norm=args.ppo_max_grad_norm,
    )
    return agent, actor_critic


def train(args, envs, encoder, agent, actor_critic, device):
    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        [encoder.feature_size],
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    if args.weights_path != "None":
        with torch.no_grad():
            obs = encoder(obs)
    else:
        obs = encoder(obs)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.ppo_use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.ppo_lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_probs, recurrent_hidden_states, actor_features, dist_entropy = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if args.weights_path != "None":
                with torch.no_grad():
                    obs = encoder(obs)
            else:
                obs = encoder(obs)

            # TODO: Check that the encoder is not updated
            # TODO: Analyze features of vae and infonce-st encoder

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_probs,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, False, args.ppo_gamma, 0.0, args.use_proper_time_limits
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if j % args.save_interval == 0 or j == num_updates - 1:
            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "ob_rms", None)],
                os.path.join(wandb.run.dir, args.env_name + ".pt"),
            )

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )
            wandb.log(
                {
                    "updates": j,
                    "total_num_steps": total_num_steps,
                    "fps": int(total_num_steps / (end - start)),
                    "episode_rewards_mean": np.mean(episode_rewards),
                    "episode_rewards_median": np.median(episode_rewards),
                    "episode_rewards_min": np.min(episode_rewards),
                    "episode_rewards_max": np.max(episode_rewards),
                    "entropy": dist_entropy,
                    "value_loss": value_loss,
                    "policy_loss": action_loss,
                }
            )


def run_rl(args):
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )
    envs = get_envs(
        env_name=args.env_name,
        seed=args.seed,
        num_processes=args.num_processes,
        num_frame_stack=args.num_frame_stack,
        downsample=not args.no_downsample,
        color=args.color,
    )
    encoder = get_encoder(args, envs.observation_space.shape, device)
    agent, actor_critic = get_agent(args, envs, encoder, device)
    train(args, envs, encoder, agent, actor_critic, device)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ["rl"]
    wandb.init(project=args.wandb_proj, tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    run_rl(args)
