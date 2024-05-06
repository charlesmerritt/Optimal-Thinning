import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from env import ForestGrowthEnv


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Simple fully connected layers for both actor and critic
        self.actor = nn.Sequential(
            nn.Linear(1, 128),  # Assuming state space is 1-dimensional
            nn.ReLU(),
            nn.Linear(128, 1),  # Assuming action space is 1-dimensional
        )
        self.critic = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = torch.sigmoid(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values


# Creating the model instance
test_model = ActorCritic()
# Test forward pass
x = torch.tensor([[100.0]])  # Example input state
action_probs, state_values = test_model(x)
print('Action probabilities:', action_probs)
print('State values:', state_values)


def ppo_update(policy, optimizer, memory, gamma, clip_param, ppo_epochs, batch_size):
    # Convert list to tensor
    states = torch.tensor(memory.states, dtype=torch.float32)
    actions = torch.tensor(memory.actions, dtype=torch.float32)
    log_probs_old = torch.tensor(memory.logprobs, dtype=torch.float32)
    rewards = torch.tensor(memory.rewards, dtype=torch.float32)
    # Normalizing rewards:
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    for _ in range(ppo_epochs):
        for index in range(0, len(states), batch_size):
            # Extract batches
            states_batch = states[index:index + batch_size]
            actions_batch = actions[index:index + batch_size]
            old_log_probs_batch = log_probs_old[index:index + batch_size]

            # Evaluate old actions and values
            new_log_probs, state_values = policy(states_batch)
            state_values = state_values.squeeze()

            # Calculate the advantage
            advantages = rewards[index:index + batch_size] - state_values.detach()

            # Calculate ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_log_probs - old_log_probs_batch.detach())

            # Actor loss using Clipped surrogate function
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = 0.5 * (rewards[index:index + batch_size] - state_values).pow(2).mean()

            # Total loss
            loss = actor_loss + critic_loss

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Clear memory after updating
    memory.clear_memory()


# Parameters for training
epochs = 10
step_per_epoch = 100
batch_size = 32
gamma = 0.99
clip_param = 0.2
ppo_epochs = 4

# Initialize policy, environment, and optimizer
policy = ActorCritic()
env = ForestGrowthEnv()
optimizer = optim.Adam(policy.parameters(), lr=0.02)
memory = Memory()

# Example training loop
for epoch in range(epochs):
    state = env.reset()
    for t in range(step_per_epoch):
        state = torch.tensor([state], dtype=torch.float32)
        action_prob, _ = policy(state)
        action = torch.distributions.Normal(action_prob, 0.1).sample().clamp(0, 1)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        # next_state, reward, done, _ = env.step(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_prob.log())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        state = next_state

        if done:
            break

    ppo_update(policy, optimizer, memory, gamma, clip_param, ppo_epochs, batch_size)

    print(f'Epoch {epoch + 1} completed')
