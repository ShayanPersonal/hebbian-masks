# hebbian masks

Update 4/10/18: Uber uses a similar idea in their "differentiable plasticity" paper which you can read here (https://arxiv.org/abs/1804.02464).

This is a experiment with what I call "hebbian masks". The idea is to prune bad connections between neurons with the hebbian learning rule. This results in smaller models and possibly has a regularizing affect.

Run with "python run.py". To augment your own modules with a hebbian mask simply wrap the HebbMask class around it: E.G. HebbMask(nn.Linear(128, 10, bias=True), ['weight'])

The gist of it is this: You store two separate variables (rather than one) for each connection in the network. One variable is the weight value you learn by gradient descent as normal. The second variable is a "hebbian" learned value learned by a hebbian learning rule. In the case of artificial neural networks, if the activation of two neurons have the same sign than the hebbian value between them increases. Otherwise it decreases. This causes anti-correlated neurons to have a low hebbian value. Weights with a low hebbian value are pruned out.

When applied to fully connected layers it seems to have no affect on learning but significantly increases the sparsity of connections. For example, I augmented the final layer of wide-resnet with a hebbian mask and it got the same validation error but with a significant amount of the synapses pruned (over 90%).

Han et al. (https://arxiv.org/abs/1506.02626) also reduce the number of parameters during training, although they use a multi-step process.

I also tried writing a version of convolutions better compatible with hebbian pruning but realized "separated convolutions" are already a thing (Google calls it "Separated convolutions" and it's used in xception for example (https://arxiv.org/abs/1610.02357), Pytorch has it partially implemented with "Grouped convolutions").
