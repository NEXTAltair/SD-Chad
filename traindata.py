#画像の特徴量をimg_features_OpenAI_CLIP_L14.npy
#画像が格納されているディレクトリの名前をmanual_scores.npyに保存するコード
import os
import clip
import torch
import glob
from PIL import Image
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser(description="train predictor")
parser.add_argument("--output", type=str, default="dataset")
parser.add_argument("--data", type=str, default="dataset", required=False)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--custom_model", type=str, default="", help="Path to the custom model")

args = parser.parse_args()

# L2正則化
def normalized(a, axis=-1, order=2):
   import numpy as np  # pylint: disable=import-outside-toplevel

   l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
   l2[l2 == 0] = 1
   return a / np.expand_dims(l2, axis)



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

data_dirctory = args.data
dir_list = glob.glob(data_dirctory+"/*")

img_features = []
manual_scores = []
c= 0
bucket_log = ""

if args.resume!="":
   img_features = np.load(f"x_{args.resume}.npy")
   manual_scores = np.load(f"y_{args.resume}.npy")
   c = img_features.shape[0]

#全体の処理時間を計測
start = time.time()

# 各ディレクトリを処理
for dir in dir_list:
   score = float(os.path.split(dir)[-1])
   files = glob.glob(dir+"/*.png") + glob.glob(dir+"/*.jpg") + glob.glob(dir+"/*.jpeg")

   if len(files) == 0:
      print(f"{dir} 内に画像がありません")
      continue
   else:
      print(f"{dir} 内の画像読み込み中...")
      
   imgnum = 0

   for img_path in files:
      try:
         image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
      except:
            continue

      with torch.no_grad():
         image_features = model.encode_image(image)

      im_emb_arr = image_features.cpu().detach().numpy()
      img_features.append(normalized(im_emb_arr))    # all CLIP embeddings are getting normalized. This also has to be done when inputting an embedding later for inference
      
      manual_scores_ = np.zeros((1, 1))
      manual_scores_[0][0] = score
      manual_scores.append(manual_scores_)



   manual_scores_ = np.zeros((1, 1))
   manual_scores_[0][0] = score
   manual_scores.append(manual_scores_)


   print(c)
   c+=1
   
img_features = np.vstack(img_features)
manual_scores = np.vstack(manual_scores)
print(img_features.shape)
print(manual_scores.shape)
np.save('img_features_OpenAI_CLIP_L14.npy', img_features)
np.save('manual_scores_.npy', manual_scores)
