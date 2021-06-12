import numpy as np
import torch
import gym
import argparse
import os
import time

import utils
import Deep_TD
import Deep_SR
import SR_DICE
import TD3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Runs policy for X episodes and returns average discounted reward
def eval_policy(policy, env, seed, eval_episodes=100):
	eval_env = gym.make(env)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	avg_discounted_reward = 0.
	timesteps = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		discount = 1.
		while not done:			
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * 0.1, size=action_dim)
			).clip(-max_action, max_action)

			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			avg_discounted_reward += discount * reward
			discount *= 0.99
			timesteps += 1

	avg_reward /= eval_episodes
	avg_discounted_reward /= (eval_episodes * timesteps)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print(f"Discounted Evaluation: {avg_discounted_reward:.5f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="SR_DICE")            # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v3")	        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=0, type=int) 	# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=50e3, type=int)  # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.2)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--random", default=0.2, type=float)        # Target network update rate
	args = parser.parse_args()

	file_name = "%s_%s_%s_%s" % (args.policy, args.env, str(args.seed), str(args.random))
	print("---------------------------------------")
	print(f"Settings: {file_name}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim, 
		"action_dim": action_dim, 
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "SR_DICE":
		ope = SR_DICE.SR_DICE(**kwargs)
	if args.policy == "Deep_SR":
		ope = Deep_SR.Deep_SR(**kwargs)
	if args.policy == "Deep_TD":
		ope = Deep_TD.Deep_TD(**kwargs)
	
	kwargs["policy_noise"] = 0.2 * max_action
	kwargs["noise_clip"] = 0.5 * max_action
	kwargs["policy_freq"] = 2
	policy = TD3.TD3(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
		
	state, done = env.reset(), False
	replay_buffer.add_start(state)

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Load behavior policy
	policy_file =  "%s_%s_%s" % ("TD3", args.env, 0)
	policy.load(f"./models/{policy_file}")
	# Evaluate behavior policy
	# eval_policy(policy, args.env, args.seed)

	# Collect data
	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps or np.random.uniform(0,1) < args.random:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 

		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
			replay_buffer.add_start(state)


	# Train and evaluate OPE
	evaluations = []
	if args.policy == "SR_DICE" or args.policy == "Deep_SR":

		print("Train Encoder-Decoder")

		for k in range(int(3e4)):
			if k % 1e3 == 0:
				print("k", k)
			ope.train_encoder_decoder(replay_buffer)

		print("Train SR")

		for k in range(int(1e5)):
			if k % 1e3 == 0:
				print("k", k)
			ope.train_SR(replay_buffer, policy.actor)

	print("Train MIS")

	for k in range(int(25e4+1)):
		ope.train_OPE(replay_buffer, policy.actor)
	
		if k % 1e3 == 0:
			print("k", k)
			evaluations.append(ope.eval_policy(replay_buffer, policy.actor))
			np.save("./results/%s" % (file_name), evaluations)