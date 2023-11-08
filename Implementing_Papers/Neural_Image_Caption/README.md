# Neural Image Caption

Based on this paper: https://arxiv.org/pdf/1411.4555.pdf \
Data: https://www.kaggle.com/datasets/adityajn105/flickr8k

#### Thoughts

Basic idea: Input Image -> CNN -> Image Embedding -> LSTM -> Image Caption

I haven't been involved in many multi-modal projects so this was fun to work on something where you are basically converting data across modalities. It's crazy to think that the image captioning task is essentially a solved problem in AI. Projects like this where you are establishing a means of converting data from one form to another will just always seem super cool to me. Definitely want to do more projects like this in the future! The implementation I developed is not exactly the same as the paper, but pretty close. I'm also using more up to date tools such as Hugging Face Transformers and Sentence Transformers which were very helpful for this task.

#### Results
Soon.

#### References
Source: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning \
Used this for the vocab class as well as some other utility functions and general problem design.

Source: https://www.youtube.com/watch?v=yCC09vCHzF8 \
"CS231n Winter 2016: Lecture 10: Recurrent Neural Networks, Image Captioning, LSTM" lecture by Andrej Karpathy that helped me to better understand some aspects of LSTMs and the network design.