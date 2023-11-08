# Neural Image Caption

Based on this paper: https://arxiv.org/pdf/1411.4555.pdf \
Data: https://www.kaggle.com/datasets/adityajn105/flickr8k

References:\
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning \
Used this for the vocab class as well as some other utility functions and general problem design.

https://www.youtube.com/watch?v=yCC09vCHzF8 \
"CS231n Winter 2016: Lecture 10: Recurrent Neural Networks, Image Captioning, LSTM" lecture by Andrej Karpathy that helped me to better understand some aspects of LSTMs and the network design.

This implementation is not exactly the same as the paper, but pretty close. I'm also using more up to date tools such as Hugging Face Transformers and Sentence Transformers which were very helpful for this task.

Basic idea: Input Image -> CNN -> Image Embedding -> LSTM -> Image Caption