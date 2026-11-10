# import os
# import csv
# from shutil import move
# from pycocotools.coco import COCO

# val_dir = "./data/images/val/"
# train_dir = "./data/images/train/"
# train_ids_csv_orig = "TrainImageIdsOriginal.csv"
# train_ids_csv = "TrainImageIds.csv"
# val_ids_csv = "ValImageIds.csv"

# split_size = 0.8
# coco = COCO('./data/annotations/captions_train2014.json')

# if not os.path.isdir(val_dir):
#     os.makedirs(val_dir)

# with open(train_ids_csv_orig, "r", newline='') as csvfile:
#     csvreader = csv.reader(csvfile, delimiter=',')
#     img_ids =list(csvreader)
# img_ids = [int(i) for i in img_ids[0]]

# # print(img_ids)

# split_id = int(split_size*len(img_ids))
# train_ids = img_ids[0:split_id]
# val_ids = img_ids[split_id:]

# with open(val_ids_csv,"w", newline='') as csvfile:
#     csvwriter = csv.writer(csvfile, dialect='excel')
#     csvwriter.writerow(val_ids)

# with open(train_ids_csv,"w", newline='') as csvfile:
#     csvwriter = csv.writer(csvfile, dialect='excel')
#     csvwriter.writerow(train_ids)
    
# for idx in val_ids:
#     path = coco.loadImgs(idx)[0]['file_name']
#     # move(train_dir+path, val_dir+path)
#     # Point to the actual raw source directory where val images were extracted
#     src_path = os.path.join("./data/val2014", path)
#     dst_path = os.path.join(val_dir, path)

#     # Check if the file exists before attempting to move it
#     if os.path.exists(src_path):
#         move(src_path, dst_path)
#     else:
#         print(f"Skipped: File not found at {src_path}")


import csv
import os
from pycocotools.coco import COCO
from shutil import move

val_dir = "./data/images/val/"
train_dir = "./data/images/train/"
train_ids_csv_orig = "TrainImageIdsOriginal.csv"
train_ids_csv = "TrainImageIds.csv"
val_ids_csv = "ValImageIds.csv"
split_size = 0.8

coco = COCO("./data/annotations/captions_train2014.json")

if not os.path.isdir(val_dir):
  os.makedirs(val_dir)
if not os.path.isdir(train_dir):
  os.makedirs(train_dir)

with open(train_ids_csv_orig, "r", newline="") as csvfile:
  csvreader = csv.reader(csvfile, delimiter=",")
  img_ids = list(csvreader)

img_ids = [int(i) for i in img_ids[0]]
split_id = int(split_size * len(img_ids))
train_ids = img_ids[0:split_id]
val_ids = img_ids[split_id:]

with open(val_ids_csv, "w", newline="") as csvfile:
  csvwriter = csv.writer(csvfile, dialect="excel")
  csvwriter.writerow(val_ids)

with open(train_ids_csv, "w", newline="") as csvfile:
  csvwriter = csv.writer(csvfile, dialect="excel")
  csvwriter.writerow(train_ids)

# 1. Move validation split images from train2014 source to val_dir
print("Moving validation images...")
for idx in val_ids:
  path = coco.loadImgs(idx)[0]["file_name"]
  # Since these IDs come from the train annotation json, they live in train2014
  src_path = os.path.join("./data/train2014", path)
  dst_path = os.path.join(val_dir, path)

  if os.path.exists(src_path):
    move(src_path, dst_path)
  else:
    print(f"Skipped (Val): File not found at {src_path}")

# 2. (Optional but recommended) Move training split images into train_dir as well
print("Moving training images...")
for idx in train_ids:
  path = coco.loadImgs(idx)[0]["file_name"]
  src_path = os.path.join("./data/train2014", path)
  dst_path = os.path.join(train_dir, path)

  if os.path.exists(src_path):
    move(src_path, dst_path)
  else:
    print(f"Skipped (Train): File not found at {src_path}")