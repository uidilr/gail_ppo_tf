import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    Policy = Policy_net('policy', env)
    BC = BehavioralCloning(Policy)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    observations = np.genfromtxt('trajectory/observations.csv')
    actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        inp = [observations, actions]

        for iteration in range(args.iteration):  # episode

            # train
            for epoch in range(args.epoch_num):
                # select sample indices in [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.minibatch_size)

                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                BC.train(obs=sampled_inp[0], actions=sampled_inp[1])

            summary = BC.get_summary(obs=inp[0], actions=inp[1])

            if (iteration+1) % args.interval == 0:
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
