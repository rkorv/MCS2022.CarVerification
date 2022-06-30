import pathlib

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize

import src.models.classifier
import src.datasets.CarsDataset
import src.utils.transforms
import src.utils.vis

from matplotlib import pyplot as plt, use

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
TARGET_SHAPE = (300, 300)
BATCH_SIZE = 8

DATASET_ROOT_DIR = "../datasets"
CHECKPOINTS_DIR = "./checkpoints"
EMBEDDING_ROOT = "./database"
SAVE_NEIGHBORS_NUM = 5

PRETRAIN = f"{CHECKPOINTS_DIR}/final.pt"

model = src.models.classifier.EffNetv2(model_name="m")
model.load_state_dict(torch.load(PRETRAIN))

model = model.eval().cuda()
model.classifier = model.classifier.float()
model.backbone = model.backbone.half()

transforms = src.utils.transforms.CarsTransforms(TARGET_SHAPE, IMAGE_MEAN, IMAGE_STD, augs=False)
full_dataset = src.datasets.CarsDataset.FullDatasetBalanced(
    DATASET_ROOT_DIR, transforms=transforms, triplet=False
)
full_loader = DataLoader(
    dataset=full_dataset, batch_size=BATCH_SIZE, num_workers=8, drop_last=False
)


def get_embeddings(model, loader):
    embeddings = np.zeros((len(loader.dataset), model.EMBEDDING_DIM), dtype=np.float32)
    all_labels = np.zeros((len(loader.dataset)), dtype=np.int)

    idx = 0
    for batch in tqdm(loader, desc="Count embeddings"):
        imgs, labels = (
            batch[0].float().half().to("cuda"),
            batch[1].long().detach().cpu().numpy(),
        )
        logits, features = model(imgs)
        features = features.detach().cpu().float().numpy()

        for feature, label in zip(features, labels):
            embeddings[idx] = feature
            all_labels[idx] = label
            idx = idx + 1

    return embeddings, all_labels


embeddings, db_labels = get_embeddings(model, full_loader)
db_embeddings = normalize(embeddings)
db_labels_list, db_labels_counts = np.unique(db_labels, return_counts=True)

save_db_embeddings = np.zeros(
    (len(db_labels_list), SAVE_NEIGHBORS_NUM, db_embeddings.shape[1]), dtype=np.float16
)

for label_num, label in tqdm(enumerate(db_labels_list)):
    label_embs = torch.from_numpy(db_embeddings[db_labels == label])
    scores = label_embs @ label_embs.T.mean(axis=1)

    if len(scores) < SAVE_NEIGHBORS_NUM:
        continue

    best_labels = scores.topk(SAVE_NEIGHBORS_NUM).indices
    save_db_embeddings[label_num] = label_embs[best_labels]

pathlib.Path(EMBEDDING_ROOT).mkdir(parents=True, exist_ok=True)

np.save(f"{EMBEDDING_ROOT}/db_embeddings", save_db_embeddings)
np.save(f"{EMBEDDING_ROOT}/db_labels", db_labels_list.astype(np.uint16))
