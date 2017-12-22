import gym
import numpy as np
import tensorflow as tf
from ppo.policy_net import Policy_net

ITERATION = int(1e4)
SAVE_INTERVAL = 1e3
MINIBATCH_SIZE = 128
EPOCH_NUM = 10


class BehavioralCloning:
    def __init__(self, Policy: Policy_net):
        self.Policy = Policy

        self.actions_expert = tf.placeholder(tf.int32, shape=[None], name='actions_expert')

        actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)

        loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss/cross_entropy', loss)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)

        self.merged = tf.summary.merge_all()

    def train(self, obs, actions):
        return tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                                      self.actions_expert: actions})

    def get_summary(self, obs, actions):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.actions_expert: actions})


def main():
    env = gym.make('CartPole-v0')
    Policy = Policy_net('policy', env)
    BC = BehavioralCloning(Policy)
    saver = tf.train.Saver(max_to_keep=20)

    observations = np.genfromtxt('trajectory/observations.csv')
    actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train/bc', sess.graph)
        sess.run(tf.global_variables_initializer())

        for iteration in range(ITERATION):  # episode

            inp = [observations, actions]

            # train
            for epoch in range(EPOCH_NUM):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=MINIBATCH_SIZE)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                BC.train(obs=sampled_inp[0],
                         actions=sampled_inp[1])

            summary = BC.get_summary(obs=inp[0],
                                     actions=inp[1])

            if iteration % SAVE_INTERVAL == 0:
                saver.save(sess, 'trained_model/bc/model.ckpt', global_step=iteration)

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
