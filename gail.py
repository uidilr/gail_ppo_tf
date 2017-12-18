#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
from ppo.policy_net import Policy_net
from ppo.ppo import PPOTrain

ITERATION = int(1e5)
GAMMA = 0.95


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        This discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a)
        """

        with tf.variable_scope('discriminator'):
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 1e-10, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 1e-10, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for policy

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.tanh, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.tanh, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    D = Discriminator(env)

    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log/train/gail.py', sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        reward = 0  # do not use rewards to update policy
        success_num = 0

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                act, _ = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if done:
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
                    saver.save(sess, 'trained_model/gail.py/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = D.get_rewards(agent_s=observations, agent_a=actions)
            gaes = np.reshape(gaes, newshape=[-1]).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            D.train(expert_s=expert_observations,
                    expert_a=expert_actions,
                    agent_s=observations,
                    agent_a=actions)

            inp = [observations, actions, gaes, rewards]

            # train policy
            PPO.assign_policy_parameters()
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
