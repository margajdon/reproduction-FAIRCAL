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
from facenet_pytorch import MTCNN, InceptionResnetV1

from arcface_model import get_arcface_model, get_input
from dependencies.mtcnn_detector import MtcnnDetector
from dir_utils import prepare_dir

import time

data_folder = "data"
embeddings_folder = "embeddings"
limit_images = None # maximum images to process

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
	choices=['mtcnn', 'vanilla'],
	default='vanilla')

parser.add_argument('--cpu', action='store_true')


def get_img_names(dataset):
	""" Get all image names (with full paths) from dataset directory """
	global limit_images

	img_names = []
	for path, subdirs, files in os.walk(os.path.join(data_folder, dataset)):
		for name in files:
			if ".jpg" in name:
				img_names.append(os.path.join(path, name))
	
	img_names = img_names[:limit_images]
	
	print(f"{len(img_names)} images found! (manual limit = {limit_images})")
	return img_names


def load_all_images(img_names):
	""" Load all images specified in img_names as np array """
	print("\nLoading images...")
	images = {}
	for i, img_name in tqdm(enumerate(img_names), total=len(img_names)):
		# alternatively allow np to load them directly
		img = Image.open(img_name)
		images[img_name] = img.copy()
		img.close()

	return images


def get_facenet_model(weights):
	""" Get pretrained facenet model """
	resnet = InceptionResnetV1(pretrained=weights, ).eval()
	return resnet


def get_bfw_img_details(img_name):
	""" Get details of image from single full-path image name"""
	category, person, image_id = img_name.replace(".jpg", "").split(os.sep)[-3:]
	data = {"category": category, "person":person, "image_id": image_id, "img_path": img_name}
	return data


def filter_by_shape(imgs, filter_img_shape=None):
	skipped = []
	# filter by shape
	print("\nFiltering images based on shape...")
	for img_name, img in tqdm(imgs.items(), total=len(imgs)):
		if img.size != filter_img_shape:
			img_details = get_bfw_img_details(img_name)
			img_details["reason"] = f"Expected shape {filter_img_shape}, got {img.size}"
			skipped.append(img_details)

	for item in skipped:
		del imgs[item["img_path"]]

	torch.cuda.empty_cache()
	return imgs, skipped


def load_images(dataset):
	img_names = get_img_names(dataset)
	return load_all_images(img_names)


def mtcnn_img_prep(all_imgs, img_names, skipped_no_face):
	print("\nMTCNN pipeline prep! (this might take a while...)")
	mtcnn = MTCNN()
	imgs_processed = mtcnn(all_imgs)
	del all_imgs
	torch.cuda.empty_cache()
	# remove images without faces
	print("\nFiltering images based on face detection...")

	for img_name, img in tqdm(zip(img_names, imgs_processed), total=len(imgs_processed)):
		if img is None:
			skipped_no_face.append(dict())
			skipped_no_face[-1].update(get_bfw_img_details(img_name))
			skipped_no_face[-1]["reason"] = "Could not find face"

	imgs_processed = [x for x in imgs_processed if x is not None]

	# cleaning up
	for item in skipped_no_face:
		img_names.remove(item["img_path"])

	return imgs_processed, img_names, skipped_no_face


def arcface_img_prep(all_imgs, img_names, skipped_no_face):
	print("\nArcface pipeline prep! (this might take a while...)")
	get_arcface_model()
	# Configure face detector
	det_threshold = [0.6, 0.7, 0.8]
	mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
	# Determine and set context
	if len(mx.test_utils.list_gpus()) == 0:
		ctx = mx.cpu()
	else:
		ctx = mx.gpu(0)

	detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True,
							 threshold=det_threshold)

	imgs_processed = []
	for img, img_name in zip(all_imgs, img_names):
		prep1 = get_input(detector, np.array(img))
		if prep1 is None:
			skipped_no_face.append(dict())
			skipped_no_face[-1].update(get_bfw_img_details(img_name))
			skipped_no_face[-1]["reason"] = "Could not find face"
		else:
			imgs_processed.append(torch.from_numpy(prep1[[2, 1, 0], :, :]))

	# cleaning up
	for item in skipped_no_face:
		img_names.remove(item["img_path"])

	return imgs_processed, img_names, skipped_no_face


def vanilla_img_prep(all_imgs, img_names, skipped_no_face):
	print("\nVanilla image prep...")
	imgs_processed = []
	to_tensor = transforms.ToTensor()
	for img in all_imgs:
		imgs_processed.append(to_tensor(img))
	return imgs_processed, img_names, skipped_no_face


def preprocess(imgs, img_prep, filter_img_shape=None):
	""" Preprocessing pipeline with MTCNN. Returns tensor of processed images, their names,
		and a data frame containing details of the skipped images. """

	skipped = []
	skipped_no_face = []
	skipped_shape = []

	if filter_img_shape:
		# filter by shape
		imgs, skipped_shape = filter_by_shape(imgs, filter_img_shape)
		skipped += skipped_shape

	# mtcnn preprocess
	img_names = list(imgs.keys())
	all_imgs = list(imgs.values())
	del imgs
	torch.cuda.empty_cache()

	img_prep_map = {
		'mtcnn': mtcnn_img_prep,
		'arcface': arcface_img_prep,
		'vanilla': vanilla_img_prep,
	}

	if img_prep is None:
		imgs_processed, img_names, skipped_no_face = vanilla_img_prep(all_imgs, img_names, skipped_no_face)
	elif img_prep in img_prep_map:
		imgs_processed, img_names, skipped_no_face = img_prep_map[img_prep](all_imgs, img_names, skipped_no_face)
	else:
		raise KeyError('Unrecognised image prep key!')

	skipped = skipped_shape + skipped_no_face
	skipped_df = pd.DataFrame(skipped)

	return imgs_processed, img_names, skipped_df


def batch(iterable, n=1):
	"""
	https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
	"""
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]


def facenet_embedding_loop(model_str, imgs, batch_size, device):
	if model_str == "facenet":
		model = get_facenet_model("vggface2")
	elif model_str == "facenet-webface":
		model = get_facenet_model("casia-webface")
	else:
		raise ValueError('Invalid model_str value!')
	model.to(device)
	embedding_list = []
	for img_batch in tqdm(batch(imgs, batch_size), total=np.ceil(len(imgs)/batch_size)):
		img_batch = torch.stack(img_batch)
		embedding_list.append(model(img_batch.to(device)).cpu().detach().numpy())
	return np.vstack(embedding_list)


def arcface_embedding_loop(model_str, imgs, batch_size, device):
	model = get_arcface_model()
	embedding_list = []
	for img_batch in tqdm(batch(imgs, batch_size), total=np.ceil(len(imgs)/batch_size)):
		img_batch = torch.stack(img_batch)
		data = mx.nd.array(img_batch)
		db = mx.io.DataBatch(data=(data,))
		model.forward(db, is_train=False)
		embedding_list.append(model.get_outputs()[0])
	return np.vstack(embedding_list)


def get_embeddings(dataset, img_names, img_prep, batch_size, model_str, device):

	filter_img_shape_map = {
		"bfw": (108, 124),
		"rfw": (400, 400)
	}
	embedding_func = {
		'facenet': facenet_embedding_loop,
		'facenet-webface': facenet_embedding_loop,
		'arcface': arcface_embedding_loop,
	}
	imgs = load_all_images(img_names)
	imgs, img_names, skipped_df = preprocess(imgs, img_prep, filter_img_shape_map[dataset])

	print("\nGenerating embeddings...")
	print("Input image shape: ", imgs[0].shape)

	embeddings = embedding_func[model_str](model_str, imgs, batch_size, device)

	details = []
	for img_name, embedding in tqdm(zip(img_names, embeddings), total=len(embeddings)):
		data = get_bfw_img_details(img_name)
		data["embedding"] = embedding
		details.append(data)

	embeddings_df = pd.DataFrame(details)
	if len(skipped_df) > 0:
		print("\nDEBUG: Both of following should be zero!")
		print(">", len(set(embeddings_df["img_path"].tolist()).intersection(skipped_df["img_path"].tolist())))
		print(">", len(set(embeddings_df["img_path"].tolist()).union(skipped_df["img_path"].tolist()))-(len(skipped_df)+len(embeddings_df)))

	return embeddings_df, skipped_df


def get_embeddings_wrapper(dataset, model_str, batch_size, img_prep, device, incremental_load=None):
	""" Get embeddings for BFW dataset for a given model. Returns embeddings+details 
		and details of skipped images. """

	img_names = get_img_names(dataset)

	if not incremental_load:
		incremental_load = len(img_names)

	total_batches = int(np.ceil(len(img_names)/incremental_load))

	edf_list = []
	sdf_list = []

	for i, img_name_batch in enumerate(batch(img_names, incremental_load)):
		print(f"> Batch {i + 1}/{total_batches}")
		embeddings_dft, skipped_dft = get_embeddings(dataset, img_name_batch, img_prep, batch_size, model_str, device)
		edf_list.append(embeddings_dft)
		sdf_list.append(skipped_dft)

	embeddings_df = pd.concat(edf_list, ignore_index=True)
	skipped_df = pd.concat(sdf_list, ignore_index=True)

	return embeddings_df, skipped_df


if __name__ == '__main__':
	start = time.time()
	args = parser.parse_args()
	
	limit_images = args.limit

	if args.cpu:
		device = torch.device("cpu")
	else:
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print(f'Using device {device}')

	print((args.model, args.batch, args.img_prep))
	embeddings_df, skipped_df = get_embeddings_wrapper(
		args.dataset, args.model, args.batch, args.img_prep, device, args.incremental
	)

	save_str = os.path.join(embeddings_folder, f"{args.model}_{args.dataset}")
	if limit_images is not None:
		save_str = save_str + f"_limited_{limit_images}"

	prepare_dir(save_str)
	embeddings_df.to_csv(f"{save_str}_embeddings.csv")
	skipped_df.to_csv(f"{save_str}_skipped.csv")

	pickle.dump(embeddings_df, open(f"{save_str}_embeddings.pk", "wb"))
	pickle.dump(skipped_df, open(f"{save_str}_skipped.pk", "wb"))

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

