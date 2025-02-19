
# Playing ATARI with Deep-Q Learning

In this relatively simple project, I implement the deep-Q network that was first suggested in the following [research paper](https://arxiv.org/abs/1312.5602).
Albeit with a few modifications as ever since the paper's publishing, a number of
issues like oscillatory learning behavior and divergence were discovered and addressed as discussed in [here](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm).

The network is trained to play [Breakout](https://ale.farama.org/environments/breakout/). But make no mistake, with
a bit more learning time, it could learn to play multiple ATARI games with a model-free agent. This was
not done here as it would involve more training time and in turn, higher monetary cost due to  GPU renting.

## ToDo
- Experiment with different hyperparameters to get reasonable performance.
- Plot results.
- Show tape of trained agent playing the game.


