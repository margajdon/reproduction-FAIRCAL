import argparse
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import torch
import pickle

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
	resnet = InceptionResnetV1(pretrained=weights).eval()
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

def preprocess_MTCNN(dataset, img_size=None, filter_img_shape=None):
	""" Preprocessing pipeline with MTCNN. Returns tensor of processed images, their names,
		and a data frame containing details of the skipped images. """

	skipped = []
	# load images
	img_names = get_img_names(dataset)
	imgs = load_all_images(img_names)
	
	# filter by shape
	imgs, skipped_shape = filter_by_shape(imgs, filter_img_shape)
	skipped += skipped_shape

	# mtcnn preprocess
	img_names = list(imgs.keys())
	all_imgs = list(imgs.values())
	del imgs
	torch.cuda.empty_cache()

	print("\nMTCNN pipeline time! (this might take a while...)")
	mtcnn = MTCNN(image_size=img_size)
	imgs_cropped = mtcnn(all_imgs)

	# remove images without faces
	skipped_no_face = []
	print("\nFiltering images based on face detection...")
	for img_name, img in tqdm(zip(img_names, imgs_cropped), total=len(imgs_cropped)):
		if img is None:
			skipped_no_face.append(dict())
			skipped_no_face[-1].update(get_bfw_img_details(img_name))
			skipped_no_face[-1]["reason"] = "Could not find face"

	imgs_cropped = [x for x in imgs_cropped if x is not None]

	# cleaning up
	for item in skipped_no_face:
		img_names.remove(item["img_path"])

	skipped = skipped_shape + skipped_no_face
	skipped_df = pd.DataFrame(skipped)

	return imgs_cropped, img_names, skipped_df

def get_bfw_embeddings(model):
	""" Get embeddings for BFW dataset for a given model. Returns embeddings+details 
		and details of skipped images. """
	imgs, img_names, skipped_df = preprocess_MTCNN("bfw", img_size=108, filter_img_shape=(108, 124))

	print("\nGenerating embeddings...")
	imgs = torch.stack(imgs)
	embeddings = model(imgs).detach().numpy()
	del imgs
	torch.cuda.empty_cache()

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

if __name__ == '__main__':
	args = parser.parse_args()
	
	limit_images = args.limit
	limit_images = 2000

	model = None
	if args.model == "facenet":
		model = get_facenet_model("vggface2")
	if args.model == "facenet-webface":
		model = get_facenet_model("casia-webface")
	
	if args.dataset == "bfw":
		embeddings_df, skipped_df = get_bfw_embeddings(model)

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
	print("*"*80)

