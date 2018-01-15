# hebbian masks and better convolutions

This is a experiment with what I call "hebbian masks". The idea is to prune unused weights with the hebbian learning rule.

When applied to fully connected layers it seems to have no affect on learning but significantly increases the sparsity of connections. For example, I augmented the final layer of wide-resnet with a hebbian mask and it got the same validation error but with a significant amount of the synapses pruned (if I remember correctly about 90%).

I also tried writing a better version of convolutions but realized that idea has been already been discovered (Google calls it "Separated convolutions" and it's used in xception for example (https://arxiv.org/abs/1610.02357), Pytorch has it partially implemented with "Grouped convolutions").
