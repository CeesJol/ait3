import simple_grid
from q_learning_skeleton import *
import gym

def act_loop(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report()
                print("state:", state)

            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(state, action, new_state, reward, done)
            state = new_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                agent.report()
                break

    env.close()


if __name__ == "__main__":
    env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    # env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")


    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount) #<- QTable
    act_loop(env, ql, NUM_EPISODES)


