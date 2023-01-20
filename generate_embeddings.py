import argparse
import os
import time
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import pickle
import mxnet as mx
import datetime as dt
from facenet_pytorch import MTCNN, InceptionResnetV1

from arcface_model import get_arcface_model, get_input
from dependencies.mtcnn_detector import MtcnnDetector

import time

from utils import batch, prepare_dir, determine_device



parser = argparse.ArgumentParser()

parser.add_argument(
	'--dataset', type=str,
	help='name of dataset whose photos should be encoded',
	choices=['rfw', 'bfw'],
	default='bfw')

parser.add_argument(
	'--model', type=str,
	help='pretrained NN model that should be used',
	choices=['facenet', 'facenet-webface', 'arcface'],
	default='facenet')

parser.add_argument(
	'--limit', type=int,
	help='maximum number of images to process',
	default=None)

parser.add_argument(
	'--batch', type=int,
	help='batch size of images passed to the model',
	default=128)

parser.add_argument(
	'--incremental', type=int,
	help='batch size of images loaded to RAM',
	default=None)

parser.add_argument(
	'--img_prep', type=str,
	help='preprocessing type applied to the images',
	choices=['mtcnn', 'vanilla', 'arcface'],
	default='vanilla')

parser.add_argument('--cpu', action='store_true')


class ImageManager:
	af_mtcnn_det = None
	limit_images = None
	data_folder = "data"
	bfw_dataset_type = 'uncropped-face-samples'
	filter_img_shape_map = {
		"bfw": (108, 124),
		"rfw": (400, 400)
	}

	def get_img_names(self, dataset):
		"""
		Get all image names (with full paths) from dataset directory
		"""
		img_names = []
		for path, subdirs, files in os.walk(os.path.join(self.data_folder, dataset)):
			if self.bfw_dataset_type not in path:
				continue
			for name in files:
				if ".jpg" in name:
					img_names.append(os.path.join(path, name))

		img_names = img_names[:self.limit_images]
		print(set([i[:25] for i in img_names]))

		print(f"{len(img_names)} images found! (manual limit = {self.limit_images})")
		return img_names

	@staticmethod
	def load_image_batch(img_names):
		"""
		Load all images specified in img_names as np array
		"""
		print("\nLoading images...")
		images = {}
		for i, img_name in tqdm(enumerate(img_names), total=len(img_names)):
			# alternatively allow np to load them directly
			img = Image.open(img_name)
			images[img_name] = img.copy()
			img.close()

		return images

	@staticmethod
	def get_img_details(img_name):
		"""
		Get details of image from single full-path image name
		"""
		category, person, image_id = img_name.replace(".jpg", "").split(os.sep)[-3:]
		data = {"category": category, "person": person, "image_id": image_id, "img_path": img_name}
		return data

	def filter_by_shape(self, imgs, dataset):
		filter_img_shape = self.filter_img_shape_map[dataset]
		skipped = []
		# filter by shape
		print("\nFiltering images based on shape...")
		# img_shapes = []
		for img_name, img in tqdm(imgs.items(), total=len(imgs)):
		# 	img_shapes.append(img.size)
		# print(set(img_shapes))
		#
		# pd.DataFrame({'shapes': img_shapes})['shapes'].value_counts()

			pass
			# if img.size != filter_img_shape:
			# 	img_details = self.get_img_details(img_name)
			# 	img_details["reason"] = f"Expected shape {filter_img_shape}, got {img.size}"
			# 	skipped.append(img_details)

		for item in skipped:
			del imgs[item["img_path"]]

		torch.cuda.empty_cache()
		return imgs, skipped

	def mtcnn_img_prep(self, all_imgs, img_names, skipped_no_face):
		print("\nMTCNN pipeline prep! (this might take a while...)")
		mtcnn = MTCNN()
		imgs_processed = mtcnn(all_imgs)
		del all_imgs
		torch.cuda.empty_cache()
		# remove images without faces
		print("\nFiltering images based on face detection...")

		for img_name, img in tqdm(zip(img_names, imgs_processed), total=len(imgs_processed)):
			if img is None:
				img_details = self.get_img_details(img_name)
				img_details["reason"] = "Could not find face"
				skipped_no_face.append(img_details)

		imgs_processed = [x for x in imgs_processed if x is not None]

		# cleaning up
		for item in skipped_no_face:
			img_names.remove(item["img_path"])

		return imgs_processed, img_names, skipped_no_face

	def arcface_img_prep(self, all_imgs, img_names, skipped_no_face):
		print("\nArcface pipeline prep! (this might take a while...)")
		detector = self.get_mtcnn_det_for_arcface()
		imgs_processed = []
		for img, img_name in zip(all_imgs, img_names):
			prep1 = get_input(detector, np.array(img))
			if prep1 is None:
				img_details = self.get_img_details(img_name)
				img_details["reason"] = "Could not find face"
				skipped_no_face.append(img_details)
			else:
				imgs_processed.append(torch.from_numpy(prep1[[2, 1, 0], :, :]))

		# cleaning up
		for item in skipped_no_face:
			img_names.remove(item["img_path"])

		return imgs_processed, img_names, skipped_no_face

	@staticmethod
	def vanilla_img_prep(all_imgs, img_names, skipped_no_face):
		print("\nVanilla image prep...")
		imgs_processed = []
		to_tensor = transforms.ToTensor()
		for img in all_imgs:
			imgs_processed.append(to_tensor(img))
		return imgs_processed, img_names, skipped_no_face

	def get_mtcnn_det_for_arcface(self):
		if self.af_mtcnn_det is None:
			# Configure face detector
			det_threshold = [0.6, 0.7, 0.8]
			mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
			# Determine and set context
			if len(mx.test_utils.list_gpus()) == 0:
				ctx = mx.cpu()
			else:
				ctx = mx.gpu(0)

			self.af_mtcnn_det = MtcnnDetector(
				model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True, threshold=det_threshold
			)
		return self.af_mtcnn_det


class EmbeddingGenerator(ImageManager):
	def __init__(self, dataset, img_prep, model_str, device, batch_size, incremental, limit_images):
		self.dataset = dataset
		self.img_prep = img_prep
		self.model_str = model_str
		self.device = device
		self.batch_size = batch_size
		self.incremental = incremental
		self.limit_images = limit_images
		self.img_prep_map = {
			'mtcnn': self.mtcnn_img_prep,
			'arcface': self.arcface_img_prep,
			'vanilla': self.vanilla_img_prep,
		}
		self.embedding_func = {
			'facenet': self.facenet_embedding_loop,
			'facenet-webface': self.facenet_embedding_loop,
			'arcface': self.arcface_embedding_loop,
		}
		if self.model_str == "facenet":
			self.model = InceptionResnetV1(pretrained="vggface2").eval()
		elif self.model_str == "facenet-webface":
			self.model = InceptionResnetV1(pretrained="casia-webface").eval()
		elif self.model_str == "arcface":
			# self.model = None
			self.model = get_arcface_model()
		else:
			raise ValueError('Unrecognised model!')

	def preprocess(self, imgs):
		""" Preprocessing pipeline with MTCNN. Returns tensor of processed images, their names,
			and a data frame containing details of the skipped images. """

		skipped = []
		skipped_no_face = []

		# Image size filtering
		imgs, skipped_shape = self.filter_by_shape(imgs, self.dataset)
		skipped += skipped_shape

		# mtcnn preprocess
		img_names = list(imgs.keys())
		all_imgs = list(imgs.values())
		del imgs
		torch.cuda.empty_cache()

		if self.img_prep in self.img_prep_map:
			imgs_processed, img_names, skipped_no_face = self.img_prep_map[self.img_prep](
				all_imgs, img_names, skipped_no_face
			)
		else:
			raise KeyError('Unrecognised image prep key!')

		skipped_df = pd.DataFrame(skipped_shape + skipped_no_face)

		return imgs_processed, img_names, skipped_df

	def facenet_embedding_loop(self, imgs):
		self.model.to(self.device)
		embedding_list = []
		for img_batch in tqdm(batch(imgs, self.batch_size), total=np.ceil(len(imgs)/self.batch_size)):
			img_batch = torch.stack(img_batch)
			embedding_list.append(self.model(img_batch.to(self.device)).cpu().detach().numpy())
		return np.vstack(embedding_list)

	def arcface_embedding_loop(self, imgs):
		embedding_list = []
		for img_batch in tqdm(batch(imgs, self.batch_size), total=np.ceil(len(imgs)/self.batch_size)):
			img_batch = torch.stack(img_batch)
			data = mx.nd.array(img_batch)
			db = mx.io.DataBatch(data=(data,))
			self.model.forward(db, is_train=False)
			embedding_list.append(self.model.get_outputs()[0].asnumpy())
		return np.vstack(embedding_list)

	def get_embedding_batch(self, img_names):

		imgs = self.load_image_batch(img_names)
		imgs, img_names, skipped_df = self.preprocess(imgs)

		print("\nGenerating embeddings...")
		print("Input image shape: ", imgs[0].shape)

		embeddings = self.embedding_func[self.model_str](imgs)

		details = []
		for img_name, embedding in tqdm(zip(img_names, embeddings), total=len(embeddings)):
			data = self.get_img_details(img_name)
			data["embedding"] = embedding
			details.append(data)

		embeddings_df = pd.DataFrame(details)
		if len(skipped_df) > 0:
			print("\nDEBUG: Both of following should be zero!")
			print(">", len(set(embeddings_df["img_path"].tolist()).intersection(skipped_df["img_path"].tolist())))
			union_len = len(set(embeddings_df["img_path"].tolist()).union(skipped_df["img_path"].tolist()))
			added_len = len(skipped_df) + len(embeddings_df)
			print(">", union_len - added_len)

		return embeddings_df, skipped_df

	def main(self):
		"""
		Get embeddings for BFW dataset for a given model. Returns embeddings+details and details of skipped images
		"""

		img_names = self.get_img_names(self.dataset)

		if not self.incremental:
			self.incremental = len(img_names)

		total_batches = int(np.ceil(len(img_names)/self.incremental))

		edf_list = []
		sdf_list = []

		for i, img_name_batch in enumerate(batch(img_names, self.incremental)):
			print(f"> Batch {i + 1}/{total_batches}")
			embeddings_dft, skipped_dft = self.get_embedding_batch(img_name_batch)
			edf_list.append(embeddings_dft)
			sdf_list.append(skipped_dft)

		embeddings_df = pd.concat(edf_list, ignore_index=True)
		skipped_df = pd.concat(sdf_list, ignore_index=True)

		return embeddings_df, skipped_df


def save_outputs(data_to_save, embeddings_folder, model, dataset, limit_images=None):
	save_str = os.path.join(embeddings_folder, f"{model}_{dataset}")
	if limit_images is not None:
		save_str = save_str + f"_limited_{limit_images}"
	prepare_dir(save_str)
	for k, df in data_to_save.items():
		df.to_csv(f"{save_str}_{k}.csv", index=False)
		pickle.dump(df, open(f"{save_str}_{k}.pk", "wb"))
	return save_str


def print_completion_info(start, save_str):
	print("\n"+"*"*80)
	print("Generated embeddings!")
	print()
	print(f"Model: {args.model}, Dataset: {args.dataset}")
	print()
	print(f"Number of embeddings: {len(embeddings_df)}")
	print(f"Saved to {save_str}_embeddings.[csv,pk]")
	print()
	print(f"Number skipped: {len(skipped_df)}")
	print(f"Saved to {save_str}_skipped.[csv,pk]")
	print(f'Time: {round(time.time() - start, 3)}s')
	print("*"*80)


if __name__ == '__main__':

	# Record time
	start = time.time()

	# Data directories
	embeddings_folder = "embeddings"

	# Parse arguments
	args = parser.parse_args()
	args.img_prep = 'arcface'
	args.model = 'arcface'
	args.incremental = 100

	# Determine the device
	device = determine_device(args.cpu)

	# Print run info
	print((args.model, args.batch, args.img_prep))

	# Generate embeddings
	embedding_generator = EmbeddingGenerator(
		args.dataset, args.img_prep, args.model, device, args.batch, args.incremental, args.limit
	)
	embeddings_df, skipped_df = embedding_generator.main()

	# Save outputs
	data_dic = {'embeddings': embeddings_df, 'skipped': skipped_df}
	save_str = save_outputs(data_dic, embeddings_folder, args.model, args.dataset, args.limit)

	# Print completion information
	print_completion_info(start, save_str)





