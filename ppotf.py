import gym
import numpy as np
import tensorflow as tf
import copy

ITERATION = int(3 * 10e5)
GAMMA = 0.95
env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Pendulum-v0')
env.seed(0)
ob_space = env.observation_space
act_space = env.action_space


class Policy_net:
    def __init__(self, name: str, temp=0.1):
        """
        :param name: string
        """
        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.nn.softmax(tf.divide(layer_3, temp))
                # self.act_probs = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss/clip'):
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

        # construct computation graph for loss of value function
        with tf.variable_scope('loss/vf'):
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf', loss_vf)

        # construct computation graph for loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entropy)

        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss  # minimize -loss == maximize loss
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, rewards, v_preds_next, gaes):
        tf.get_default_session().run([self.train_op], feed_dict={self.Policy.obs: obs,
                                                                 self.Old_Policy.obs: obs,
                                                                 self.actions: actions,
                                                                 self.rewards: rewards,
                                                                 self.v_preds_next: v_preds_next,
                                                                 self.gaes: gaes})

    def get_summary(self, obs, actions, rewards, v_preds_next, gaes):
        return tf.get_default_session().run([self.merged], feed_dict={self.Policy.obs: obs,
                                                                      self.Old_Policy.obs: obs,
                                                                      self.actions: actions,
                                                                      self.rewards: rewards,
                                                                      self.v_preds_next: v_preds_next,
                                                                      self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(run_policy_steps - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes


Policy = Policy_net('policy')
Old_Policy = Policy_net('old_policy')
PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log/', sess.graph)
    sess.run(tf.global_variables_initializer())
    obs = env.reset()
    reward = 0
    success_num = 0

    for iteration in range(ITERATION):  # episode
        observations = []
        actions = []
        v_preds = []
        v_preds_next = []
        rewards = []
        done = False
        run_policy_steps = 0
        while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
            run_policy_steps += 1
            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
            act, v_pred = Policy.act(obs=obs, stochastic=True)

            # print(Policy.get_action_prob(obs=obs))

            act = np.asscalar(act)
            v_pred = np.asscalar(v_pred)

            # print(v_pred)
            # print(obs[0, 0], act)

            observations.append(obs)
            actions.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)

            next_obs, reward, done, info = env.step(act)

            if done:
                v_preds_next = v_preds[1:] + [0]  # v_preds of next states of terminate state is 0
                obs = env.reset()
                reward = -1
                break
            else:
                obs = next_obs

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                           , iteration)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                           , iteration)
        if sum(rewards) >= 195:
            success_num += 1
            if success_num >= 100:
                saver.save(sess, './model/model.ckpt')
                print('model saved. done.')
                break
        else:
            success_num = 0

        gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

        # convert list to numpy array for feeding tf.placeholder
        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)
        rewards = np.array(rewards).astype(dtype=np.float32)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()

        PPO.assign_policy_parameters()

        inp = [observations, actions, rewards, v_preds_next, gaes]

        # train
        for epoch in range(4):
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            PPO.train(obs=sampled_inp[0],
                      actions=sampled_inp[1],
                      rewards=sampled_inp[2],
                      v_preds_next=sampled_inp[3],
                      gaes=sampled_inp[4])

        summary = PPO.get_summary(obs=inp[0],
                                  actions=inp[1],
                                  rewards=inp[2],
                                  v_preds_next=inp[3],
                                  gaes=inp[4])[0]

        writer.add_summary(summary, iteration)
    writer.close()
