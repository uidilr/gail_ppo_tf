import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain

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
        reward = 0
        success_num = 0

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=False)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has value 0
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            # end condition of test
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
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

            inp = [observations, actions, rewards, v_preds_next, gaes]

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      rewards=inp[2],
                                      v_preds_next=inp[3],
                                      gaes=inp[4])[0]

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
