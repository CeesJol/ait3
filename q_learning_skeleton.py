
NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1




class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"



    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass




    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        pass



    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """
        pass



    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")








        
