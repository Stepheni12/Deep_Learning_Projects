# Neural Image Caption

Based on this paper: https://arxiv.org/pdf/1411.4555.pdf \
Data: https://www.kaggle.com/datasets/adityajn105/flickr8k

#### Thoughts

Basic idea: Input Image -> CNN -> Image Embedding -> LSTM -> Image Caption

I haven't been involved in many multi-modal projects so this was fun to work on something where you are basically converting data across modalities. It's crazy to think that the image captioning task is essentially a solved problem in AI. Projects like this where you are establishing a means of converting data from one form to another will just always seem super cool to me. Definitely want to do more projects like this in the future! The implementation I developed is not exactly the same as the paper, but pretty close. I'm also using more up to date tools such as Hugging Face Transformers and Sentence Transformers which were very helpful for this task.

#### Results
It was cool sampling the same image on the model at various different points throughout training to see how it was improving. This could've been trained longer, but due to a lack of resources I cut it early. If I get a chance to train it longer in the future I'll get a longer training run in.

These are some random images I pulled from the internet. There's a mix of good and bad results, overall I'm happy with how these turned out.

Example Images:
"a woman and a girl on a beach" \
![kids standing in front of water at beach watching people in the water](https://github.com/Stepheni12/Deep_Learning_Projects/blob/main/Implementing_Papers/Neural_Image_Caption/example_images/beach.png?raw=true)

"a small brown dog is standing on its hind legs" \
![a light brown cat sitting on a leather stool](https://github.com/Stepheni12/Deep_Learning_Projects/blob/main/Implementing_Papers/Neural_Image_Caption/example_images/cat.png?raw=true)

"a dog in a red collar is standing in the snow" \
![a dog in a red collar or jacket sitting in the snow](https://github.com/Stepheni12/Deep_Learning_Projects/blob/main/Implementing_Papers/Neural_Image_Caption/example_images/dog.png?raw=true)

"a man in a red shirt and hat talks to a man in a black shirt" \
![three woman and a man running together](https://github.com/Stepheni12/Deep_Learning_Projects/blob/main/Implementing_Papers/Neural_Image_Caption/example_images/run.png?raw=true)

#### References
Source: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning \
Used this for the vocab class as well as some other utility functions and general problem design.

Source: https://www.youtube.com/watch?v=yCC09vCHzF8 \
"CS231n Winter 2016: Lecture 10: Recurrent Neural Networks, Image Captioning, LSTM" lecture by Andrej Karpathy that helped me to better understand some aspects of LSTMs and the network design.