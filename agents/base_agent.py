

import numpy as np

class BaseAgent:
    def __init__(self, name, env, seed=0, reward_type=None):
        self.name = name
        self.env = env
        self.seed = seed
        self.training = True
        self.reward_type = reward_type  # store reward type if needed


    def set_eval(self):
        self.training = False

    def set_train(self):
        self.training = True

    def act(self, obs, exploration=False):
        raise NotImplementedError

    def eval(self, env, exploration=False, render=False):
   # """
   # Evaluate agent in given environment.
   # Returns:
   #     total_reward (float),
   #     infos (list of dicts with date, weights, values, etc.),
   #     final_port_value (float)
   # """
        self.set_eval()
        obs = env.reset()
        done = False
        rewards = []
        port_value = []
        infos = []   # ✅ collect all step info

        while not done:
        # ✅ Convert dict observation into (prices, weights)
            if isinstance(obs, dict):
                prices = obs.get("prices")
                weights = obs.get("weights", np.zeros(len(env.stock_names) + 1))
                obs = (prices, weights)

        # Choose action safely
            action = self.act(obs, exploration=exploration)

        # Step environment
            next_obs, reward, done, info = env.step(action)

        # ✅ Ensure next_obs is converted too
            if isinstance(next_obs, dict):
                prices = next_obs.get("prices")
                weights = next_obs.get("weights", np.zeros(len(env.stock_names) + 1))
                next_obs = (prices, weights)

            obs = next_obs
            rewards.append(reward)
            port_value.append(info.get("port_value_new", env.init_port_value))

        # ✅ ensure required fields for evaluator
            if "date" not in info and hasattr(env, "market"):
                step_idx = len(infos)
                if step_idx < len(env.market.date_list):
                    info["date"] = env.market.date_list[step_idx]
            if "weights_new" not in info:
                info["weights_new"] = action
            if "port_value_old" not in info:
                info["port_value_old"] = port_value[-1]

            infos.append(info)

        return sum(rewards), infos, port_value[-1] if port_value else env.init_port_value

