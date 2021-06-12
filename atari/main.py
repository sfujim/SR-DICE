import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import glob

import utils
import DDQN
import Deep_TD
import Deep_SR
import SR_DICE


def interact_with_environment(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	kwargs = {
		"is_atari": is_atari,
		"num_actions": num_actions,
		"state_dim": state_dim,
		"device": device,
		"discount": parameters["discount"],
		"optimizer": parameters["optimizer"],
		"optimizer_parameters": parameters["optimizer_parameters"],
		"polyak_target_update": parameters["polyak_target_update"],
		"target_update_frequency": parameters["target_update_freq"],
		"tau": parameters["tau"],
		"initial_eps": parameters["initial_eps"],
		"end_eps": parameters["end_eps"],
		"eps_decay_period": parameters["eps_decay_period"],
		"eval_eps": parameters["eval_eps"]
	}

	# Initialize and load policy
	policy = DDQN.DDQN(**kwargs)

	if args.generate_buffer: 
		for result in glob.glob(f"./models/{args.env}/Q*"):
			policy.load(result)
			print("loaded")

	evaluations = []

	state, done = env.reset(), False
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	low_noise_ep = np.random.uniform(0,1) < args.low_noise_p
	replay_buffer.add_start(state)

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# If generating the buffer, episode is low noise with p=low_noise_p.
		# If policy is low noise, we take random actions with p=eval_eps.
		# If the policy is high noise, we take random actions with p=rand_action_p.
		# if args.generate_buffer:
		if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(state), eval=True)

		# Perform action and log results
		next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			low_noise_ep = np.random.uniform(0,1) < args.low_noise_p
			replay_buffer.add_start(state)

		# Evaluate episode
		if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


def train_ope(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	kwargs = {
		"is_atari": is_atari,
		"num_actions": num_actions,
		"state_dim": state_dim,
		"device": device,
		"discount": parameters["discount"],
		"optimizer": parameters["optimizer"],
		"optimizer_parameters": parameters["optimizer_parameters"],
		"polyak_target_update": parameters["polyak_target_update"],
		"target_update_frequency": parameters["target_update_freq"],
		"tau": parameters["tau"],
		"initial_eps": parameters["initial_eps"],
		"end_eps": parameters["end_eps"],
		"eps_decay_period": parameters["eps_decay_period"],
		"eval_eps": parameters["eval_eps"]
	}

	# Make DICE here
	if args.algo =="SR_DICE":
		ope = SR_DICE.SR_DICE(**kwargs)
	if args.algo =="Deep_SR":
		ope = Deep_SR.Deep_SR(**kwargs)
	if args.algo =="Deep_TD":
		ope = Deep_TD.Deep_TD(**kwargs)

	# make DDQN here
	policy = DDQN.DDQN(**kwargs)

	# bad code but it works
	for result in glob.glob(f"./models/{args.env}/Q*"):
		policy.load(result)
		print("loaded")

	policy = policy.Q

	# Load replay buffer	
	replay_buffer.load(f"./buffers/{buffer_name}")

	evaluations = []

	if args.algo == "SR_DICE" or args.algo == "Deep_SR":

		for _ in range(int(3e4)):
			ope.train_encoder_decoder(replay_buffer)
		
		# Train psi
		for _ in range(int(1e5)):
			ope.train_SR(replay_buffer, policy)
	
	#Train DICE
	for k in range(int(25e4+1)):
		if k % 1e3 == 0:
			ev = ope.eval_policy(replay_buffer, policy)
			evaluations.append(ev)
			np.save(f"./results/{args.algo}_{setting}", evaluations)

		ope.train_OPE(replay_buffer, policy)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=100):
	eval_env, is_atari, _, _ = utils.make_env(env_name, atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	avg_discounted_reward = 0.
	discount = 1.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		discount = 1.

		while not done:
			
			if np.random.uniform(0,1) < 0.1:
			 	action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state), eval=True)

			state, reward, done, info = eval_env.step(action)
		
			if is_atari:
				reward = info[0]
				done_float = info[1]
				
			avg_reward += reward
			avg_discounted_reward += discount * reward
			discount *= 0.99

	avg_reward /= eval_episodes
	avg_discounted_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print(f"Discounted Evaluation: {avg_discounted_reward:.10f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"eps": 0.00015
			#"lr": 3e-4
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}

	regular_parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	env_list = [
		"BeamRiderNoFrameskip-v0",
		"KrullNoFrameskip-v0",
		"BreakoutNoFrameskip-v0",
		"AsterixNoFrameskip-v0",
		"PongNoFrameskip-v0",
	]

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", default="SR_DICE") 
	parser.add_argument("--env", default=0, type=int)
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
	parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
	parser.add_argument("--low_noise_p", default=0.0, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.1, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
	parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
	args = parser.parse_args()
	
	args.env = env_list[args.env]

	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training OPE, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	parameters = atari_parameters if is_atari else regular_parameters

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	if args.train_behavioral or args.generate_buffer:
		interact_with_environment(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)
	else:
		train_ope(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)