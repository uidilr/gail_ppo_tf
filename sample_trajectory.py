import gym
import numpy as np
import os
from ppo.policy_net import Policy_net
import tensorflow as tf

ITERATION = int(1)
GAMMA = 0.95


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

            np.savetxt('trajectory/observations.csv', observations, delimiter=',')
            np.savetxt('trajectory/actions.csv', actions, delimiter=',')


if __name__ == '__main__':
    main()
    x = np.genfromtxt('trajectory/observations.csv', delimiter=',')
    y = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    print(x)
    print(y)
