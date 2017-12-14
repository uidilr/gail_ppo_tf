import gym
import numpy as np
import os
from ppo.policy_net import Policy_net
import tables
import tensorflow as tf


ITERATION = int(3)
GAMMA = 0.95


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    :return:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'ppo/trained_model/model.ckpt')
        obs = env.reset()

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)

                next_obs, reward, done, info = env.step(act)

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            open_file_and_save('trajectory/observations.csv', observations)
            open_file_and_save('trajectory/actions.csv', actions)


if __name__ == '__main__':
    main()
    x = np.genfromtxt('trajectory/observations.csv')
    y = np.genfromtxt('trajectory/actions.csv', dtype=np.int)
    print(len(x))
    print(len(y))
