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
from facenet_pytorch import MTCNN, InceptionResnetV1


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

parser.add_argument('--mtcnn', action='store_true')
parser.add_argument('--no-mtcnn', dest='mtcnn', action='store_false')

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
	data = {"category":category, "person":person, "image_id":image_id, "img_path":img_name}
	return data

def filter_by_shape(imgs, filter_img_shape=None):
	skipped = []
	# filter by shape
	print("\nFiltering images based on shape...")
	for img_name, img in tqdm(imgs.items(), total=len(imgs)):
		if img.size != filter_img_shape:
			skipped.append(dict())
			skipped[-1].update(get_bfw_img_details(img_name))
			skipped[-1]["reason"] = f"Expected shape {filter_img_shape}, got {img.size}"

	for item in skipped:
		del imgs[item["img_path"]]

	torch.cuda.empty_cache()
	return imgs, skipped

def preprocess(dataset, device, use_MTCNN=True, filter_img_shape=None):
	""" Preprocessing pipeline with MTCNN. Returns tensor of processed images, their names,
		and a data frame containing details of the skipped images. """

	skipped = []
	skipped_no_face = []
	skipped_shape = []
	# load images
	img_names = get_img_names(dataset)
	imgs = load_all_images(img_names)
	
	if filter_img_shape is not None:
		# filter by shape
		imgs, skipped_shape = filter_by_shape(imgs, filter_img_shape)
		skipped += skipped_shape

	# mtcnn preprocess
	img_names = list(imgs.keys())
	all_imgs = list(imgs.values())
	del imgs
	torch.cuda.empty_cache()
	imgs_processed = []
	if use_MTCNN:
		print("\nMTCNN pipeline time! (this might take a while...)")
		mtcnn = MTCNN(device=device)
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

	else:
		print("\nSkipping MTCNN...")
		imgs_processed = []
		for img in all_imgs:
			imgs_processed.append(transforms.ToTensor()(img).to(device))

	skipped = skipped_shape + skipped_no_face
	skipped_df = pd.DataFrame(skipped)

	return imgs_processed, img_names, skipped_df

def batch(iterable, n=1):
	# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_bfw_embeddings(model, batch_size, use_MTCNN, device):
	""" Get embeddings for BFW dataset for a given model. Returns embeddings+details 
		and details of skipped images. """
	imgs, img_names, skipped_df = preprocess("bfw", device, use_MTCNN=use_MTCNN, filter_img_shape=(108, 124))

	print("\nGenerating embeddings...")
	
	# Getting these embeddings to cuda gives only slow down
	embeddings = None
	for img_batch in tqdm(batch(imgs, batch_size), total=np.ceil(len(imgs)/batch_size)):
		img_batch = torch.stack(img_batch)
		if embeddings is None:
			embeddings = model(img_batch.to(device)).cpu().detach().numpy()
		else:
			embeddings = np.vstack([embeddings, model(img_batch.to(device)).cpu().detach().numpy()])
		
		del img_batch
		torch.cuda.empty_cache()

	details = []
	for img_name, embedding in zip(img_names, embeddings):
		data = get_bfw_img_details(img_name)
		data["embedding"] = embedding
		details.append(data)

	embeddings_df = pd.DataFrame(details)
	if len(skipped_df) > 0:
		print("\nDEBUG: Both of following should be zero!")
		print(">", len(set(embeddings_df["img_path"].tolist()).intersection(skipped_df["img_path"].tolist())))
		print(">", len(set(embeddings_df["img_path"].tolist()).union(skipped_df["img_path"].tolist()))-(len(skipped_df)+len(embeddings_df)))

	return embeddings_df, skipped_df

def get_rfw_embeddings(model, batch_size, use_MTCNN, device):
	""" Get embeddings for BFW dataset for a given model. Returns embeddings+details 
		and details of skipped images. """
	imgs, img_names, skipped_df = preprocess("rfw", device, use_MTCNN=use_MTCNN, filter_img_shape=(400, 400))

	print("\nGenerating embeddings...")
	
	# Getting these embeddings to cuda gives only slow down
	embeddings = None
	for img_batch in tqdm(batch(imgs, batch_size), total=np.ceil(len(imgs)/batch_size)):
		img_batch = torch.stack(img_batch)
		if embeddings is None:
			embeddings = model(img_batch.to(device)).cpu().detach().numpy()
		else:
			embeddings = np.vstack([embeddings, model(img_batch.to(device)).cpu().detach().numpy()])
		
		del img_batch
		torch.cuda.empty_cache()

	details = []
	for img_name, embedding in zip(img_names, embeddings):
		data = get_bfw_img_details(img_name)
		data["embedding"] = embedding
		details.append(data)

	embeddings_df = pd.DataFrame(details)
	if len(skipped_df) > 0:
		print("\nDEBUG: Both of following should be zero!")
		print(">", len(set(embeddings_df["img_path"].tolist()).intersection(skipped_df["img_path"].tolist())))
		print(">", len(set(embeddings_df["img_path"].tolist()).union(skipped_df["img_path"].tolist()))-(len(skipped_df)+len(embeddings_df)))

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

	model = None
	if args.model == "facenet":
		model = get_facenet_model("vggface2")
	if args.model == "facenet-webface":
		model = get_facenet_model("casia-webface")
	
	model = model.to(device)

	if args.dataset == "bfw":
		embeddings_df, skipped_df = get_bfw_embeddings(model, args.batch, args.mtcnn, device)
	else:
		embeddings_df, skipped_df = get_rfw_embeddings(model, args.batch, args.mtcnn, device)

	save_str = os.path.join(embeddings_folder, f"{args.model}_{args.dataset}")
	if limit_images is not None:
		save_str = save_str + f"_limited_{limit_images}"

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

