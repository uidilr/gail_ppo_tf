import gym
import numpy as np
import tensorflow as tf
from ppotf import Policy_net, PPOTrain

ITERATION = int(3 * 10e5)
GAMMA = 0.95


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'model/model.ckpt')
        obs = env.reset()

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, _ = Policy.act(obs=obs, stochastic=False)

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


if __name__ == '__main__':
    main()
