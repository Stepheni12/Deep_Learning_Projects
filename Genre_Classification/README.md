# Genre Classification

Data: https://www.kaggle.com/datasets/ishikajohari/imdb-data-with-descriptions/data

I really wouldn't recommend this dataset as it became quite frustrating to work with at times. There's quite a few issues as a result of pulling movie/show information from two different sources. I laid out a lot of the issues with the dataset in this post: https://www.kaggle.com/datasets/ishikajohari/imdb-data-with-descriptions/discussion/443834

Despite the dataset issues I still want to circle back and clean this project up to a put a final pin in it.

With that said I still learned so much from this project. If it wasn't obivous from the notebook this project was shortly after I discovered Andrej Karpathy's 'A Recipe for Training Neural Networks' it was one of the better spelled-out somewhat step-by-step attempts to implementing neural nets which is a pretty foreign concept in this field. It seems as though every machine learning practitioner has a different method of solving problems using deep learning and I've been researching a ton of them lately(Josh Tobin has a good one). I personally like the idea of a well-formulated approach which you can apply to any project so I figured I'd give his recipe a shot.

I stuck to his method pretty strictly and I definitely was a fan of the many training checks he has in place, many of which I hadn't thought to implement previously. I also think he has model diagnostics laid out very nicely. Although it took some time to get pytorch to cooperate with all of what I was trying to keep track of throughout training. I learned all about pytorch hooks and how poor the documentation on them is.

In general I've been trying to come up with a more strategic and repeatable approach to solving these deep learning problems regardless of the dataset. I want to be as efficient as possible. No problem is the same so there are some adjustments that need to be made each time, but I've definitely streamlined my time to a solution now that I've gone down this path. Through the help of utilizing model diagnostics and running some basic tests to ensure the network is training properly as well as following the sound advice of Andrej Karpathy along the way I'm definitely getting closer to the efficiency I want to be at.