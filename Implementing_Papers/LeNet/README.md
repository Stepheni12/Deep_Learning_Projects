# Gradient-Based Learning Applied to Document Recognition (LeNet Paper)

Based on this paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf \
Data: MNIST (There's python code in the notebook to pull the data)

#### Thoughts
I really just wanted to do this to say I did it. This was such an influential paper, it was pretty cool reading a paper from the late 90s on machine learning and seeing what they did and didn't know (full disclosure, I didn't read the whole thing). The paper itself is quite different than the norms of today. It's almost 50 pages which is a bit much, however, in terms of explaining the network architecture and the reasoning behind certain design decisions this paper is pretty incredible. It definitely includes much more details than what you see today. I feel as though this is something that's missing from many papers released today, obviously the architectures are much more in-depth today, but I feel like more effort in explaining the architectures and design principles would go a long way.

#### Implementation Notes
There are a couple areas where my implementation differs from the original architecture:
- The paper uses a different type of pooling than what we are accustomed to today. In their version there is learnable parameters in the pooling layers which isn't something I've heard of, probably wouldn't be too difficult to implement. I'm sure there's probably something online already. Got curious and did some quick googling: https://github.com/mbrukman/reimplementing-ml-papers/blob/main/lenet/subsampling.py 
- In the C3 layer of paper, they implement the filters in an interested way in which certain layers of the output only see certain layers of the input. Also something I hadn't heard of before. They do explain the reasoning in the paper and it seemed like a decent idea. I wonder if anyone today is still doing any convolutions like this.
- They used a custom scaled tanh activation.
- They utilized a custom learning rate scheduler.

Maybe one day I'll come back to this and implement some of these differences.

#### PyTorch Lightning Implementation
For my first go at lightning I decided to just convert this notebook into a typical lightning setup. First thoughts are that I can definitely see the potential, it's pretty great to be honest. Definitely much simpler than the typical pytorch code you would write, it's more understandable, and the reusability factor is exponentially improved. If you used this setup across all your projects it would operate similarly (to a lesser extent) to how hugging face does with all their pre-trained models, where you can switch in and out various aspects of different models and combine them in a variety of ways. One downside I've already noticed is that the documentation kind of stinks, but for now, I'm excited about learning more about all the features.

#### Results
Train Accuracy: 99% \
Test Accuracy: 98% \
2500 training steps with batches of 128 \
SGD with learning rate of 0.1

#### References
Source: https://animatedai.github.io/ \
Not entirely related to this specific task, but this was just a really cool set of convolution animations I came across which I think would be beneficial to people trying to understand convolutions for the first time.