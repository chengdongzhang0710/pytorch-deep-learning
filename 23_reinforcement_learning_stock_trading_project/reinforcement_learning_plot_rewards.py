import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
args = parser.parse_args()
a = np.load(f'rl_trader_rewards/{args.mode}.npy')
print(f'average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}')

plt.figure()
if args.mode == 'train':
    plt.plot(a)  # show the training process
else:
    plt.hist(a, bins=20)  # show a histogram of rewards
plt.title(args.mode)
plt.savefig(f'{args.mode}.png')
