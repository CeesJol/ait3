import numpy as np
import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import debug_utils

NUM_EPISODES = 500
MAX_EPISODE_LENGTH = 30000

RMSIZE = 10000 # replay memory size
BATCH_SIZE = 256  # size of replay memory batch (= the number of updates for each real step)

# consistent with Shiverma's code:
DEFAULT_DISCOUNT = 0.99
EPSILON = 1
LEARNINGRATENET = 0.0001  # QNET


# TODO coding exercise 3: implement experience replay
class ReplayMemory(object):
    # experiences = []
    # ReplayMemory should store the last "size" experiences
    # and be able to return a randomly sampled batch of experiences
    def __init__(self, size):
        self.experiences = []
        self.size = size
        self.counter = 0

    # Store experience in memory
    def store_experience(self, prev_obs, action, observation, reward, done):
        experience = (prev_obs, action, observation, reward, done)

        if (len(self.experiences) < self.size):
            self.experiences.append(experience)
        else:
            self.experiences[self.counter % self.size] = experience

        self.counter += 1

    # Randomly sample "batch_size" experiences from the memory and return them
    def sample_batch(self, batch_size):
        if batch_size > len(self.experiences):
            raise Exception('Not enough experiences')
        return random.sample(self.experiences, batch_size)


# DEBUG=True
DEBUG = False


class QNet(nn.Module):

    def __init__(self, num_a, obs_shape, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATENET):
        nn.Module.__init__(self)

        self.discount = discount
        self.learning_rate = learning_rate

    def init_optimizer(self):
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.9)

    def obs_to_tensor(self, obs):
        """ QNet uses pytorch, and hence all observations need to be wrapped as tensors """
        if not isinstance(obs, torch.Tensor):
            if np.isscalar(obs):
                obs = [obs]
            obs = torch.Tensor(obs)
        return obs

    def max_Q_value(self, observation, batch_size=1):
        observation = self.obs_to_tensor(observation)
        Qs = self.forward(observation) #<- this should feed in the input
        #print ("QNet::max_Q_value: Qs: %s", Qs)

        if batch_size > 1:
            v,_ = Qs.max(dim=1)
        else: 
            v = Qs.max()
        v = v.detach().numpy()
        #print ("... Vs: %s" % v)
        return v

    def argmax_Q_value(self, observation):
        """ observation is a single observation - does not work for batch """
        observation = self.obs_to_tensor(observation)
        t_Qs = self.forward(observation) #<- this should feed in the input

        #NOOO! this will *not* randomly break ties(!)...
        # m = np.argmax(self.data[observation, :]) 
        #instead:
        Qs = t_Qs.detach().numpy()
        if DEBUG:
            print("argmax_Q_value: Qs: %s" % Qs)
        m = np.random.choice(np.flatnonzero(Qs == Qs.max()))
        return m

    def get_Q(self, o, a, batch_size=1):
        observation = self.obs_to_tensor(o)
        Qs = self.forward(observation)
        if batch_size > 1:
            q = Qs[range(batch_size), a]
        else:
            q = Qs[a]

        return q

    def single_Q_update(self, prev_observation, action, observation, reward, done):
        """ action and observation need to be in the format that QNet was constructed for.
            I.e., if observation is a discrete variable (with say N values=states), but QNet
            is working on one-hot vectors (of length N), then observation needs to be such a
            one-hot vector.

            QNet is not responsible for conversion
        """
        t_observation = self.obs_to_tensor(observation)
        t_prev_observation = self.obs_to_tensor(prev_observation)

        if done:
            future_val = 0 
        else:
            future_val = self.max_Q_value(t_observation)        ##<<- this evaluates the QNet
        # We just evaluated the Qnet for the next-stage variables, but of course... the effect of the Qnet 
        # parameters on the *next-stage* value is ignored by Q-learning. 
        # (residual gradient algorithms do takes this into account, but 
        #   formally need 2 successor state samples)
        # So... we need to reset the gradients. (otherwise they accumulate e.g., see; 
        # https://medium.com/@zhang_yang/how-pytorch-tensors-backward-accumulates-gradient-8d1bf675579b)
        self.zero_grad()

        #S&B v1, p211
        # theta' := theta + learning_rate * TD     (* eligibility, which we ignore)
        t_predict = self.get_Q(t_prev_observation, action)  ##<<- this evaluates the QNet
        t_predict.backward()  # computes grad_theta Q(s,a)
        target = reward + self.discount * future_val
        td = target - t_predict.detach().numpy()

        # now update all the parameters
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1) # <--- apply gradient clipping to avoid exploding gradients...
            param.data.add_(self.learning_rate * td * param.grad.data)

        self.zero_grad()

        predict = t_predict.detach().numpy()
        new_q = self.get_Q(t_prev_observation, action).detach().numpy()
        self.zero_grad()

        debug_utils.debug_q_update(prev_observation, action, observation, reward, done, predict, self.discount, future_val,
                                   target, td, new_q)

    def batch_Q_update(self, obs, actions, next_obs, rewards, dones):
        # print("dones = " , dones)
        batch_size = len(dones)
        v_next_obs = self.max_Q_value(next_obs, batch_size)
        not_dones = 1 - dones
        fut_values = self.discount * v_next_obs * not_dones
        targets = rewards + fut_values

        self.zero_grad()
        q_pred = self.get_Q(obs, actions, batch_size)
        loss = self.loss_fn(q_pred, torch.tensor(targets, dtype=torch.float))
        if DEBUG:
            print("q_pred:    %s" % q_pred)
            print("loss:      %s" % loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QNet_MLP(QNet):
    def __init__(self, num_a, obs_shape, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATENET):
        super().__init__(num_a, obs_shape, discount=discount, learning_rate=learning_rate)
        self.init_network(obs_shape, num_a)
        self.init_optimizer()

    def init_network(self, obs_shape, num_a):
        num_in = np.prod(obs_shape)
        print("QNet_MLP initialization: num_in=%s, obs_shape=%s" % (num_in, obs_shape))
        ### MLP
        HIDDEN_NODES1 = 150
        HIDDEN_NODES2 = 120
        self.fc1 = nn.Linear(num_in, HIDDEN_NODES1)  # 6*6 from image dimension
        self.fc2 = nn.Linear(HIDDEN_NODES1, HIDDEN_NODES2)
        self.fc3 = nn.Linear(HIDDEN_NODES2, num_a)  #4 outputs, the Q-values for the 4 actions
        #nn.init.constant_(self.fc1.weight, 0.0)
        #nn.init.constant_(self.fc2.weight, 0.0)
        #nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """ This assumes x to be a tensor """
        debug_utils.assert_isinstance(x, torch.Tensor)
        ### MLP:
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        ###----
        return x


class QLearner(object):
    def __init__(self, env, q_function, discount=DEFAULT_DISCOUNT, rm_size=RMSIZE):
        self.env = env
        self.Q = q_function
        self.rm = ReplayMemory(rm_size)  # replay memory stores (a subset of) experience across episode
        self.discount = discount

        self.epsilon = EPSILON
        self.epsilon_min = .01
        self.epsilon_decay = .98

        self.batch_size = BATCH_SIZE

        self.name = "agent1"
        self.episode = 0
        self.cum_r = 0 	    #cumulative reward in current episode
        self.tot_r = 0      #cumulative reward in lifetime
        self.stage = 0      #the time step, or 'stage' in this episode
        self.tot_stages = 0 #total time steps in lifetime


    def reset_episode(self, initial_obs):
        self.last_obs = initial_obs
        self.tot_r += self.cum_r #store the reward of the previous episode
        self.cum_r = 0           #reset cumulative reward for new episode
        self.dis_r = 0           #discounted cum. reward
        self.tot_stages += self.stage
        self.stage = 0           #reset the time step, or 'stage' in this episode
        self.episode += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decay epsilon

    def process_experience(self, action, observation, reward, done):
        prev_observation = self.last_obs
        self.cum_r += reward
        self.dis_r += reward * (self.discount ** self.stage)
        self.stage += 1
        self.Q.single_Q_update(prev_observation, action, observation, reward, done)
        self.last_obs = observation
        self.rm.store_experience(prev_observation, action, observation, reward, done)
        if self.tot_stages > 10 * self.batch_size:
            updateBatch = self.rm.sample_batch(self.batch_size)
            obs = []
            actions = []
            next_obs = []
            rewards = []
            dones = np.array([])
            for x in updateBatch:
                obs.append(x[0])
                actions.append(x[1])
                next_obs.append(x[2])
                rewards.append(x[3])
                dones = np.append(dones, x[4])
            #print("dones = ", dones)
            #dones = np.invert(dones)
            self.Q.batch_Q_update(obs, actions, next_obs, rewards, dones)
            # and update the network using this batch (batch_Q_update)
            # def batch_Q_update(self, obs, actions, next_obs, rewards, dones):
            # experience = {prev_obs, action, observation, reward, done}


    def select_action(self):
        """select an action based on self.last_obs

           (In general we might select actions on more general information... i.e., last_obs could
            be generalized to last_internal_state )
        """
        if random.random() < self.epsilon:
            action = random.randint(0, self.env.action_space.n - 1)
            if DEBUG:
                print("select_action_random used")
        else:
            #obs = self.gym2qnet_observation(self.last_obs) #XXX
            obs = self.last_obs
            action = self.Q.argmax_Q_value(obs)
            if DEBUG:
                print("select_action_greedy used")

        return action

    def report(self):
        name = self.name
        print("---")
        print("%s: episode: %d" % (name, self.episode))
        print("%s: stage:   %d" % (name, self.stage))
        print("%s: totals stages:   %d" % (name, self.tot_stages))
        print("%s: epsilon: %f" % (name, self.epsilon))
        print("%s: cum_r:   %s" % (name, self.cum_r))
        print("%s: dis_r:   %s" % (name, self.dis_r))
        mean_r_this_ep = self.cum_r / self.stage if self.stage > 0 else "undef"
        mean_r = self.tot_r / self.tot_stages if self.tot_stages > 0 else "undef"
        mean_r_ep = self.tot_r / self.episode if self.episode > 0 else "undef"
        print("%s: mean r in this episode:  %s" % (name, mean_r_this_ep))
        print("%s: mean r in lifetime:      %s" % (name, mean_r))
        print("%s: mean return per episode:   %s" % (name, mean_r_ep))
