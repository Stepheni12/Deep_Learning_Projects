import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import requests
import gzip
import os
import hashlib

class MnistDataset(Dataset):

	def __init__(self, images_path, labels_path):
		self.images = np.load(images_path)
		self.labels = np.load(labels_path)
		self.length = len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]

		image = torch.from_numpy(image)
		label = torch.from_numpy(np.array(label))

		return image, label

	def __len__(self):
		return self.length


class MnistDataModule(pl.LightningDataModule):

	def __init__(self, data_dir, batch_size, num_workers):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.images_path = os.path.join(self.data_dir, "images.npy")
		self.labels_path = os.path.join(self.data_dir, "labels.npy")

	def prepare_data(self):
		X_1 = self._fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28)) / 255.0
		Y_1 = self._fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
		X_2 = self._fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28)) / 255.0
		Y_2 = self._fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

		images = np.concatenate([X_1, X_2], axis=0, dtype=np.float32)
		labels = np.concatenate([Y_1, Y_2], axis=0)

		images = self._transform(images)

		try:
			np.save(self.images_path, images)
			np.save(self.labels_path, labels)
			print("Data Saved.")
		except Exception as e:
			print(f"Save failed with error: {e}")

	def setup(self, stage):
		mnist = MnistDataset(self.images_path, self.labels_path)

		train_size = int(0.8 * len(mnist))
		val_size = int(0.1* len(mnist))
		test_size = len(mnist) - train_size - val_size

		self.train_dataset, self.val_dataset, self.test_dataset = random_split(mnist, 
																[train_size, val_size, test_size])

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers)

	def _fetch(self, url):
	    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
	    if os.path.isfile(fp):
	        with open(fp, "rb") as f:
	            dat = f.read()
	    else:
	        with open(fp, "wb") as f:
	            dat = requests.get(url).content
	            f.write(dat)
	    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

	def _transform(self, arr):
		resized_lst = []

		for img in arr:
			resized_lst.append(Image.fromarray(img).resize((32,32)))

		ret = np.stack(resized_lst, axis=0)
		ret = ret.reshape(-1, 1, 32, 32)

		return ret