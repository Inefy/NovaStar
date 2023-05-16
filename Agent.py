import torch.optim as optim

class DQNAgent(base_agent.BaseAgent):
    def __init__(self, action_size, state_size, hidden_size=128, lr=1e-3):
        super(DQNAgent, self).__init__()
        
        self.action_size = action_size
        self.state_size = state_size

        self.model = SimpleNet(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.numpy())

    def update_model(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        reward = torch.tensor(reward).float().unsqueeze(0)
        action = torch.tensor(action).long().unsqueeze(0)

        curr_Q = self.model(state)[0][action]
        next_Q = self.model(next_state).max(1)[0].detach()
        target_Q = reward + (0.99 * next_Q) * (1 - done)

        loss = self.criterion(curr_Q, target_Q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
