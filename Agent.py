class DQNAgent(base_agent.BaseAgent):
    def __init__(self, action_size, state_size, hidden_size=128, lr=1e-3, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, target_update=10, memory_size=10000):
        super(DQNAgent, self).__init__()

        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        self.model = SimpleNet(self.state_size, self.action_size)
        self.target_model = SimpleNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(memory_size)

    def get_epsilon(self, steps_done):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * steps_done / self.epsilon_decay)

    def get_action(self, state, steps_done):
        epsilon = self.get_epsilon(steps_done)
        if np.random.rand() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.numpy())
        else:
            return random.randrange(self.action_size)

    def update_model(self):
    if len(self.memory) < self.batch_size:
        return
    state, action, reward, next_state, done = self.memory.sample(self.batch_size)
    state = torch.from_numpy(state).float()
    next_state = torch.from_numpy(next_state).float()
    reward = torch.tensor(reward).float()
    action = torch.tensor(action).long()
    done = torch.tensor(done).bool()

    curr_Q = self.model(state).gather(1, action.unsqueeze(1))
    next_action = self.model(next_state).max(1)[1] 
    next_Q = self.target_model(next_state).gather(1, next_action.unsqueeze(1)).squeeze(1).detach()
    expected_Q = reward + self.gamma * next_Q * (1 - done)

    loss = self.criterion(curr_Q, expected_Q.unsqueeze(1))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


        curr_Q = self.model(state).gather(1, action.unsqueeze(1))
        next_Q = self.target_model(next_state).max(1)[0].detach()
        expected_Q = reward + self.gamma * next_Q * (1 - done)

        loss = self.criterion(curr_Q, expected_Q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def run(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            steps_done = 0
            while not done:
                action = self.get_action(state, steps_done)
                next_state, reward, done, _ = env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                self.update_model()
                state = next_state
                steps_done += 1

                if done:
                    self.update_target_model()
                    print(f"Episode {episode} finished after {steps_done} timesteps")

            if episode % self.target_update == 0:
                self.update_target_model()

            if episode % 100 == 0:
                torch.save(self.model.state_dict(), f"model_{episode}.pth")

        print("Training complete.")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

