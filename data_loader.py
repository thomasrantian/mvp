import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class cabinet(Dataset):
	def __init__(self, root, train = False, transforms = None):

		self.root = root
		self.transforms = transforms
		self.train = train


		self.train_data = [file for file in os.listdir(root) if "data_batch" in file]
		self.test_data = [file for file in os.listdir(root) if "test_batch" in file]

		self.data_files = self.train_data if self.train else self.test_data
		self.images = []
		self.labels = []
		
		self.load_data()


	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.images[idx]
		image = Image.fromarray(image)

		label = self.labels[idx]

		if self.transforms:
			image = self.transforms(image)

		return image, label


	def load_data(self):

		for file in self.data_files:
			file_path = os.path.join(self.root, file)
			sample = self.read_file(file_path)
			self.images.append(sample["data"])
			self.labels.extend(sample["labels"])


		self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
		self.images = self.images.transpose((0, 2, 3, 1))

	def read_file(self, filename):
		with open(filename, "rb") as f:
			f = pickle.load(f, encoding = "latin1")
		return f
