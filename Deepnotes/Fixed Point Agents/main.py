import torch
import torch.nn as nn

class FixedPointLayer(nn.Module):
    def __init__(self, output_features, tolerance=1e-4, max_iterations=50):
        super(FixedPointLayer, self).__init__()
        self.linear = nn.Linear(output_features, output_features, bias=False)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def forward(self, x):
        # Initialize the output to be zero.
        z = torch.zeros_like(x)
        self.iterations = 0
        # Iterate until convergence.
        while self.iterations < self.max_iterations:
            z_next = torch.tanh(self.linear(z) + x)
            self.error = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            if self.error < self.tolerance:
                break 
        return z
    
    def extra_repr(self):
        return 'tolerance={}, max_iterations={}'.format(self.tolerance, self.max_iterations)
    
class Transformers(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, num_iterations=1):
        super(Transformers, self).__init__()
        self.num_iterations = num_iterations
        self.transformer = nn.Sequential()
        for i in range(self.num_iterations):
            self.transformer.add_module(str(i), FixedPointLayer(output_features))
        self.linear = nn.Linear(input_features, output_features)
        
    def forward(self, x):
        z = torch.tanh(self.linear(x))
        for i in range(self.num_iterations):
            z = self.transformer[i](z)
        return z
 
class Actor(nn.Module):
    def __init__(self, state_space, action_space, seed, layers=[256, 256]):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        self.layers = layers
        self.layers[0] = state_space
        self.layers.append(action_space)
        
        self.transformer = Transformers(input_features=state_space, output_features=layers[0], hidden_features=state_space)
        
        self.hidden = nn.ModuleList()
        for i in range(len(self.layers) - 2):
            self.hidden.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            setattr(self, 'hidden_{}'.format(i), nn.Linear(self.layers[i], self.layers[i + 1]))
        
        self.output = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, 'hidden_{}'.format(len(self.layers) - 2), nn.Linear(self.layers[-2], self.layers[-1]))
    
    def forward(self, x):
        z = self.transformer(x)
        for i in range(len(self.layers) - 2):
            z = self.hidden[i](z)
            z = torch.tanh(z)
        z = self.output(z)
        return torch.tanh(z) * self.action_space

    
class Critic(nn.Module):
    def __init__(self, state_space, action_space, seed, layers=[256, 256]):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_space = action_space
        self.layers = layers
        self.layers[0] = state_space + action_space
        
        self.transformer = Transformers(input_features=state_space + action_space, output_features=layers[0], hidden_features=state_space + action_space)
        
        self.hidden = nn.ModuleList()
        for i in range(len(self.layers) - 2):
            self.hidden.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            setattr(self, 'hidden_{}'.format(i), nn.Linear(self.layers[i], self.layers[i + 1]))
        
        self.output = nn.Linear(self.layers[-2], 1)
        setattr(self, 'hidden_{}'.format(len(self.layers) - 2), nn.Linear(self.layers[-2], 1))
    
    def forward(self, x, a):
        z = torch.cat((x, a), 1)
        z = self.transformer(z)
        for i in range(len(self.layers) - 2):
            z = self.hidden[i](z)
            z = torch.tanh(z)
        z = self.output(z)
        return z
class DDPG():
    def __init__(self, state_space, action_space, random_seed, epsilon=1.0, epsilon_decay=1e-6, epsilon_min=0.01, gamma=0.99, tau=1e-3, lr_a=1e-4, lr_c=1e-3, weight_decay=0):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        
        self.actor = Actor(state_space, action_space, random_seed).to(device)
        self.actor_target = Actor(state_space, action_space, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a, weight_decay=weight_decay)
        
        self.critic = Critic(state_space, action_space, random_seed).to(device)
        self.critic_target = Critic(state_space, action_space, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=weight_decay)
        
        # Copy the weights from actor and critic to their respective target networks.
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor(state).cpu().data.numpy()
        if np.random.rand() < self.epsilon:
            action += np.random.normal(0, 0.1)
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        done = torch.from_numpy(done).float().to(device)
        
        # Compute Q values for current states using the critic network.
        q_values = self.critic(state, action)
        
        # Compute the next actions using the actor target network and compute their Q values using the critic target network.
        next_actions = self.actor_target(next_state)
        q_next = self.critic_target(next_state, next_actions)
        
        # Compute the expected Q values.
        q_expected = reward + self.gamma * q_next * (1 - done)
        
        # Compute the critic loss.
        critic_loss = F.mse_loss(q_values, q_expected)
        
        # Minimize the critic loss.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute new actions using the actor network and compute their Q values using the updated critic network.
        new_actions = self.actor(state)
        q_predicted = self.critic(state, new_actions)
        
        # Compute and minimize the actor loss. 
        actor_loss = -q_predicted.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update the target networks by slowly copying the weights from the original networks.
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        
        # Update epsilon value to gradually reduce exploration over time.
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
agent = DDPG(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0], random_seed=10)
def ddpg(n_episodes=1000, max_steps=1000):
    scores = []
    scores_window = deque(maxlen=100)
    for i in range(1, n_episodes+1):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Uncomment this line to run the trained agent with noise turned off.
            # action = agent.act(state)
            action = agent.act(state) + np.random.normal(0, 0.1)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        scores.append(total_reward)
        scores_window.append(total_reward)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            break
    return scores
scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()

# If you want to run the trained agent with noise turned off, uncomment the code below.
# agent = DDPG(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0], random_seed=10)
# agent.actor.load_state_dict(torch.load('checkpoint_actor.pth'))
# agent.critic.load_state_dict(torch.load('checkpoint_critic.pth'))
# state = env.reset()
# for t in range(200):
#     action = agent.act(state) 
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break 
# env.close()
