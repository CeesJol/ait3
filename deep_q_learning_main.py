import gym

from deep_q_learning_skeleton import *

# Set to true if you want the agent to take into account the remaining time
# (an episode automatically stops after 1000 timesteps)
timeHorizon = True

def act_loop(env, agent, num_episodes):
    for episode in range(num_episodes):
        observation = env.reset()
        if timeHorizon:
            observation = np.append(observation,1)
        agent.reset_episode(observation)

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        # for t in range(MAX_EPISODE_LENGTH):
        t = 0
        while True:
            t += 1
            if renderit:
                env.render()
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report()
                print("obs:", observation)

            action = agent.select_action()
            observation, reward, done, info = env.step(action)
            if timeHorizon:
                timeRemaining = (1000 - t) / 1000 # goes from 1 at first timestep to 0 at last timestep
                observation = np.append(observation, timeRemaining)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(action, observation, reward, done)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                agent.report()
                break

    env.close()


if __name__ == "__main__":
    # from def_env import env  #<- defines env
    env = gym.make('LunarLander-v2')
    print("action space:", env.action_space)
    print("observ space:", env.observation_space)

    num_a = env.action_space.n
    shape_o = env.observation_space.shape
    if timeHorizon:
        shape_o = (9,)

    qn = QNet_MLP(num_a, shape_o)

    discount = DEFAULT_DISCOUNT

    # ql = QLearner(env, qn, discount) #<- QNet

    # TODO: Coding exercise 4: target network
    target_qn = QNet_MLP(num_a, shape_o)
    target_qn.load_state_dict(qn.state_dict())
    ql = QLearner(env, qn, target_qn, discount)  # <- QNet

    act_loop(env, ql, NUM_EPISODES)
