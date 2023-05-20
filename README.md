# NovaStar

This repository contains an implementation of an AI agent that uses Deep Q-Learning Network (DQN) to play StarCraft II. 

## Files

### replaybuffer.py

This file contains the implementation of the Prioritized Experience Replay (PER) buffer, which is an enhancement over the standard experience replay buffer. It stores the agent's experiences and samples them with priority, helping the agent to learn from the most informative transitions.

### agent.py

This file contains the implementation of the DQN agent. The agent is implemented as a subclass of the base StarCraft II agent class, and it uses a Dueling DQN architecture for its value function approximation. The agent interacts with the environment, stores its experiences in the replay buffer, samples from it to update its policy, and periodically synchronizes its target and behavior networks.

### sc2envwrapper.py

This file contains a wrapper for the StarCraft II environment. It simplifies the interaction between the agent and the environment and calculates the reward the agent receives.

## Usage

1. Install the necessary dependencies (e.g., PySC2, TensorFlow).
2. Initialize the environment and the agent.
3. Call the agent's `run` method to start the training process.

```python
env = SC2EnvWrapper(...)
agent = DQNAgent(...)
agent.run(env, num_episodes)
```

4. The agent's model parameters are saved every 100 episodes. You can load the parameters to resume training or to evaluate the agent's performance.

```python
agent.load_model(path_to_model)
```

## Notes

- This implementation uses a Dueling DQN and Prioritized Experience Replay, two enhancements over the standard DQN, to improve the learning efficiency and stability.
- The reward calculation is a simple example and can be modified based on your specific requirements.
- The action space and observation space are not fully utilized. You can modify the `get_action` method of the agent and the `step` method of the environment wrapper to use more actions and observations.

### network.py

This file contains the definition of the Dueling DQN architecture. The network has two streams, one for estimating the state value function and the other for estimating the state-dependent action advantages. The final Q-value is calculated by combining the outputs of these two streams.

### base_agent.py

This is the base class for our DQN agent. The DQN agent inherits from this class to gain access to basic functionalities such as initializing the agent, taking actions in the environment, and receiving rewards.

### requirements.txt

This file lists the Python dependencies necessary for running the code. Before running the code, install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

## Customization

This AI is designed to be flexible and easy to customize. Here are a few ways I plan to potentially modify this project:

- **Different Rewards**: The reward function in `sc2envwrapper.py` can be customized to better suit your requirements.
- **Different Network Architecture**: The DQN agent uses a Dueling DQN architecture, but you could replace it with another type of function approximator, such as a convolutional neural network for image-based observations.
- **Different Replay Buffer**: The agent uses a prioritized replay buffer, which can be replaced with a standard replay buffer or another type of replay buffer.
- **Different Exploration Strategy**: The agent uses epsilon-greedy exploration, but you could replace it with another type of exploration strategy, such as softmax exploration or upper confidence bound (UCB) exploration.

## Potential Improvements

There are several ways I plan to potentially improve the performance of the AI:

- **Tune Hyperparameters**: The performance of DQN can be highly sensitive to its hyperparameters. You could potentially improve the AI's performance by tuning the hyperparameters, such as the learning rate, batch size, discount factor, and the parameters of the epsilon-greedy exploration strategy.
- **Use a Larger Replay Buffer**: DQN can benefit from a larger replay buffer, which allows it to learn from a larger set of experiences.
- **Use a More Complex Reward Function**: The reward function in `sc2envwrapper.py` is quite simple. A more complex reward function that provides more granular feedback could potentially improve the AI's performance.
- **Use More Actions and Observations**: The AI currently uses a very limited set of actions and observations. Using a larger set of actions and observations could allow the AI to learn more complex strategies.

