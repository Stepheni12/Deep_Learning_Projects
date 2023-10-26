This was my first attempt at an object detection model. This dataset only contained a single object which was cars, however, certain images had multiple instances of cars.
I learned a ton throughout this project. I utilized pretrained pytorch models in order to implement transfer learning to fine-tune the model to my dataset. The way the pretrained
pytorch object detection datasets expect the input took some getting used to, but it is clearly well thought out and scales effectively to object detection models of many classes. 
While it is effective there is definitely a bit of a learning curve to figuring out how to prepare your dataset into a form that the model likes. This project also gave me a much
stronger understanding of creating Datasets and Dataloaders in pytorch and why they are so useful. I want to expand on what I learned from this project by gaining a deeper understanding
of the actual steps of the object detection model.


This project is based on a kaggle dataset which you can find here: https://www.kaggle.com/datasets/sshikamaru/car-object-detection

