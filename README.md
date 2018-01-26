# hebbian masks and better convolutions

Run with "python run.py"

This is a experiment with what I call "hebbian masks". The idea is to prune unused weights with the hebbian learning rule.

When applied to fully connected layers it seems to have no affect on learning but significantly increases the sparsity of connections. For example, I augmented the final layer of wide-resnet with a hebbian mask and it got the same validation error but with a significant amount of the synapses pruned (if I remember correctly over 90%).

Han et al. (https://arxiv.org/abs/1506.02626) also reduce the number of parameters during training, although they use a multi-step process.

I also tried writing a version of convolutions compatible with hebbian pruning but realized that idea has been already been discovered (Google calls it "Separated convolutions" and it's used in xception for example (https://arxiv.org/abs/1610.02357), Pytorch has it partially implemented with "Grouped convolutions").