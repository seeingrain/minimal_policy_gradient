import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
    def save(self, suffix):
        torch.save(self, f'policy_{suffix}.pt')
        print(f'Policy saved to policy_{suffix}.pt')



def train():
    # Plot duration curve: 
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Parameters
    num_episode = 20000
    batch_episodes = 10
    learning_rate = 0.01

    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500
    pygame.display.set_caption(f'Training... Press S to save checkpoint, Q to quit')

    policy_net = PolicyNet()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    played_episodes = 0
    continuous_good = False

    while(True):
        batch_trajectories = []

        '''
            play 'batch_size' episodes, and record data into 'batch_trajectories[]'
        '''
        for _ in range(batch_episodes):
            trajectory_dict = {'states': [], 'actions': [], 'rewards': []}
            
            # play a single episode
            state = env.reset()
            state = torch.from_numpy(state)

            for step in count():
                # sample an actionfrom policy network
                distribution = Categorical(policy_net(state))
                action = distribution.sample()
                action = action.data.numpy().astype(int)

                # step the game
                next_state, reward, done, _ = env.step(action)
                env.render(mode='rgb-array') # change to human to visualize realtime video, at the cost of training speed

                # record 'state', 'action', 'reward' into 'trajectory_dict'
                trajectory_dict['states'].append(state)
                trajectory_dict['actions'].append(action)
                trajectory_dict['rewards'].append(reward)

                state = torch.from_numpy(next_state)

                # check for keyboard event
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            policy_net.save(played_episodes)
                        elif event.key == pygame.K_q:
                            return

                if done:
                    episode_durations.append(len(trajectory_dict['states']))
                    plot_durations()

                    # save policy if past 20 episodes are all good (can play for >= 450 steps)
                    if len(episode_durations) >= 20:
                        if continuous_good == False:
                            if all(x >= 450 for x in episode_durations[-20:]):
                                policy_net.save(f'episode{played_episodes}-duration{episode_durations[-1]}')
                                continuous_good = True
                        else:
                            if not all(x >= 450 for x in episode_durations[-20:]):
                                continuous_good = False
                    
                    break

            batch_trajectories.append(trajectory_dict) # push this trajectory into batch
            played_episodes += 1


        '''
            Update policy by batched trajectories
        '''
        optimizer.zero_grad()

        # standardize rewards
        batch_rewards = []
        for trajectory in batch_trajectories:
            batch_rewards.append(sum(trajectory['rewards']))
        batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-7)
        assert(not torch.isnan(torch.tensor(batch_rewards)).any())

        for index, episode in enumerate(batch_trajectories):
            episode_steps = len(episode['states'])
            episode_total_reward = batch_rewards[index]

            for step in range(episode_steps):
                played_state = episode['states'][step]
                played_action = torch.from_numpy(episode['actions'][step])

                distribution = Categorical(policy_net(played_state))

                # only accumulate the gradient, but do NOT update yet
                loss = -(distribution.log_prob(played_action) * episode_total_reward)
                loss.backward()

        # Update the policy by accumulated gradient from the entire batch
        optimizer.step()

        if played_episodes >= num_episode:
            policy_net.save(played_episodes)
            break

    input("Press Enter to continue...")



def eval(policy_file_name):
    pygame.init()
    pygame.display.set_caption(f'{policy_file_name}, Press Q to quit')
    screen = pygame.display.set_mode((400, 300))
    font = pygame.font.Font(None, 24)

    with torch.no_grad():
        policy_net = torch.load(policy_file_name)

        env = gym.make('CartPole-v0')
        env._max_episode_steps = 10000
        
        state = env.reset()
        state = torch.from_numpy(state)

        steps = 0

        while(True):
            steps += 1

            distribution = Categorical(policy_net(state))
            action = distribution.sample()
            action = action.data.numpy().astype(int)

            next_state, _, done, _ = env.step(action)
            env.render(mode='human')

            # show steps in game
            score_text = font.render(f"{steps} steps", True, (0, 0, 0))
            screen.blit(score_text, (10, 10))  # (10, 10)是得分文本左上角的位置
            pygame.display.flip()

            state = torch.from_numpy(next_state)

            # check for quit
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return

            if done:
                print(f'Episode finished after {steps} steps')
                env.reset()
                steps = 0



'''
    not implemented yet.
'''
# def human_play():
#     env = gym.make('CartPole-v0')
#     env._max_episode_steps = 10000

#     env.render(mode='human')

#     while(True):
#         probs = policy_net(state)
#         m = Bernoulli(probs)
#         action = m.sample()

#         action = action.data.numpy().astype(int)
#         next_state, reward, done, _ = env.step(action)
#         env.render(mode='human')

#         state = next_state
#         state = torch.from_numpy(state).float()
#         state = Variable(state)



if __name__ == '__main__':
    train()
    # eval('policy_episode277-duration500.pt')
    # human_play()
