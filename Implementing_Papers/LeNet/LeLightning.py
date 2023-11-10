import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchmetrics import Metrics
import os

class LeLightning(pl.LightningModule):
    def __init__(self):
        super(LeNet, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.sm = torch.nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        self.c1 = torch.nn.Conv2d(1, 6, 5)
        self.s2 = torch.nn.AvgPool2d(2, 2)
        self.c3 = torch.nn.Conv2d(6, 16, 5)
        self.s4 = torch.nn.AvgPool2d(2,2)
        self.c5 = torch.nn.Conv2d(16, 120, 5)
        self.f6 = torch.nn.Linear(120, 84)
        self.f7 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.tanh(self.c1(x))
        x = self.s2(x)
        x = F.tanh(self.c3(x))
        x = self.s4(x)
        x = F.tanh(self.c5(x))
        x = x.view(-1, 120)
        x = F.tanh(self.f6(x))
        x = self.f7(x)
        return x

    # Created this common step because this is a simpler problem where train, val, test is all the same
    # For more complex tasks you can define each of these differently!
    def _common_step(self, batch, batch_idx):
        x, y = batch
    	logits = self.forward(x)
    	loss = self.loss_func(logits, y)
    	return loss, logits, y

    def training_step(self, batch, batch_idx):
    	loss, logits, y = self._common_step(batch, batch_idx)
    	accuracy = self.accuracy(self.sm(logits), y)
    	self.log_dict({'train_loss', loss, 'train_accuracy': accuracy},
    				  on_step=False, on_epoch=True, prog_bar=True)
    	return {'loss': loss, 'logits': logits, 'y': y}

    def validation_step(self, batch, batch_idx):
    	loss, logits, y = self._common_step(batch, batch_idx)
    	self.log('val_loss', loss)
    	return loss

    def test_step(self, batch, batch_idx):
    	loss, logits, y = self._common_step(batch, batch_idx)
    	self.log('test_loss', loss)
    	return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
    	logits = self.forward(x)
    	probs = self.sm(logits)
    	preds = torch.argmax(probs)
    	return preds

    def configure_optimizers(self): # And schedulers
    	return torch.optim.SGD(model.parameters(), lr=0.1)


    # Also going to define a bunch of other functions you can define with the lightning module
    def training_epoch_end(self, outputs): #
    	pass

    def training_step_end(self, outputs):
    	pass

class MnistDataset(Dataset):

	def __init__(self, images_path, labels_path):
		self.images = np.load(images_path)
		self.labels = np.load(labels_path)
		self.length = len(self.images)

	def __getitem__(self, idx):
		image = self.data[idx]
		label = self.label[idx]

		image = torch.from_numpy(image)
		label = torch.from_numpy(label)

		return image, label

	def __len__(self):
		return self.length


class MnistDataModule(pl.LightningDataModule):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.images_path = os.path.join(self.data_dir, "images.npy")
		self.labels_path = os.path.join(self.data_dir, "labels.npy")


	def prepare_data(self):
		X_1 = _fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28)) / 255.0
		Y_1 = _fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
		X_2 = _fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28)) / 255.0
		Y_2 = _fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

		images = np.concatenate([X_1, X_2], axis=0)
		labels = np.concatenate([Y_1, Y_2], axis=0)

		np.save(images_path, images)
		np.save(labels_path, labels)


	def setup(stage):
		mnist = MnistDataset(images_path, labels_path)

		train_size = int(0.8 * len(mnist))
		val_size = int(0.1* len(mnist))
		test_size = len(mnist) - train_size - val_size

		self.train_dataset 

	def train_dataloader(self):
		pass

	def val_dataloader(self):
		pass

	def test_dataloader(self):
		pass

	def _fetch(self, url):
	    import requests, gzip, os, hashlib, numpy
	    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
	    if os.path.isfile(fp):
	        with open(fp, "rb") as f:
	            dat = f.read()
	    else:
	        with open(fp, "wb") as f:
	            dat = requests.get(url).content
	            f.write(dat)
	    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


model = LeLightning()

trainer = pl.Trainer() # Lots of cool functionality with all the different params for Trainer class
trainer.fit(model, train_loader, val_loader)