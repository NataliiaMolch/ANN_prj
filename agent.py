class RLAgent(object):
    
    def __init__(self, n_obs, policy_learning_rate, value_learning_rate, 
                 discount, baseline=None, fileNamePolicy=None, fileNameValue=None):

        #We need the state and action dimensions to build the network
        self.n_obs = n_obs  
        self.n_act = 1
        
        self.gamma = discount
        
        self.use_baseline = baseline is not None
        self.use_adaptive_baseline = baseline == 'adaptive'

        #Fill in the rest of the agent parameters to use in the methods below
        
        # TODO
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate

        #These lists stores the observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        #Build the keras network
        self.fileNamePolicy = fileNamePolicy
        self.fileNameValue = fileNameValue
        self.model = self._build_network(lr = self.policy_learning_rate)
        self.value_model = self._build_network(lr = self.value_learning_rate, last_activation = 'selu',  loss = 'mse')
        
        
    def observe(self, state, action, reward):
        """ This function takes the observations the agent received from the environment and stores them
            in the lists above. """
        # TODO
        self.episode_observations.append(state)
        self.episode_actions.append(action)
        #print(action)
        self.episode_rewards.append(reward)
            
    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode, then optionally apply a baseline. Hint: work backwards."""
        # TODO
        returns = []
        tmp = 0.

        for rwd in self.episode_rewards[::-1]:
            tmp *= self.gamma
            tmp += rwd
            returns.append(tmp)
            
        if self.use_baseline and not self.use_adaptive_baseline:
            return np.array(list(map(lambda r : r - returns[-1]*self.gamma - self.episode_rewards[0], returns[::-1])))
        elif self.use_adaptive_baseline:
            values = self.value_model.predict(np.stack(self.episode_observations, axis = 0))
            return np.array(returns[::-1]) - values[:,0]
        else:
            return np.array(returns[::-1])
        
    def get_target_values(self):
        returns = []
        tmp = 0.

        for rwd in self.episode_rewards[::-1]:
            tmp *= self.gamma
            tmp += rwd
            returns.append(tmp)
        return returns[::-1]
    
    def _build_network(self, lr, last_activation = 'sigmoid', loss = 'binary_crossentropy'):
        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""
        # TODO
        model = Sequential()
        init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        model.add(Dense(32, activation='relu', input_dim = 7, kernel_initializer = init))
        model.add(Dense(32, activation='relu', kernel_initializer = init))
        model.add(Dense(32, activation='relu', kernel_initializer = init))
        model.add(Dense(1, activation=last_activation))
        model.compile(optimizer = keras.optimizers.Adam(lr=lr),
                           loss= loss)
        return model
        

    def decide(self, state):
        """ This function feeds the observed state to the network, which returns a distribution
            over possible actions. Sample an action from the distribution and return it."""
        # TODO
        return int(np.random.binomial(1, self.model.predict(state[np.newaxis,:])))

    def train(self):
        """ When this function is called, the accumulated observations, actions and discounted rewards from the
            current episode should be fed into the network and used for training. Use the _get_returns function 
            to first turn the episode rewards into discounted returns. """
        # TODO
        returns = self._get_returns()
        gamma = np.ones_like(returns) * self.gamma

        self.model.train_on_batch(np.stack(self.episode_observations, axis = 0),
                                  self.episode_actions,
                                  sample_weight = returns)
        if self.use_adaptive_baseline:
            self.value_model.train_on_batch(np.stack(self.episode_observations, axis = 0),
                                  self.get_target_values(),
                                  sample_weight = returns * gamma)
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        