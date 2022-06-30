import os

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional

import src.datasets.CarsDataset
import src.utils.transforms
import src.models.classifier


EMBEDDING_DIM = 2048
NUM_CLASSES = 5540

DATASET_ROOT_DIR = "../datasets"
DATABASE_ROOT = "./database"
SUBMITS_DIR = "./"

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
SUBMIT_NAME = "submission"

NEIGHBORS_COUNT_BATCH = 1024

MODELS = [
    {
        "model": "checkpoints/final.pt",
        "target_shape": (300, 300),
        "bs": 16,
        "version": "m",
        "flip": False,
        "use_db": False,
        "neightbors_num": 5,
        "weight": 1,
    },
    # {
    #     "model": "checkpoints/final.pt",
    #     "target_shape": (300, 300),
    #     "bs": 16,
    #     "version": "m",
    #     "flip": True,
    #     "use_db": False,
    #     "neightbors_num": 5,
    #     "weight": 1,
    # },
    # {
    #     "model": "ckpts/large_epoch4_256.pt.sd",
    #     "target_shape": (256, 256),
    #     "bs": 8,
    #     "version": "l",
    #     "flip": False,
    #     "use_db": False,
    #     "neightbors_num": 2,
    #     "weight": 0.75,
    # },
]


def load_db():
    db_embeddings = np.load(f"{DATABASE_ROOT}/db_embeddings.npy").astype(np.float32)
    db_embeddings = db_embeddings.reshape(db_embeddings.shape[0] * db_embeddings.shape[1], -1)
    db_labels = np.load(f"{DATABASE_ROOT}/db_labels.npy").astype(np.uint16)
    db_one_hot_labels = np.zeros((len(db_labels), db_labels.max() + 1), dtype=np.float32)
    db_one_hot_labels[np.arange(len(db_labels)), db_labels] = 50
    db_one_hot_labels = db_one_hot_labels.repeat(5, axis=0)
    return db_embeddings, db_one_hot_labels


if any([m["use_db"] for m in MODELS]):
    db_embeddings, db_one_hot_labels = load_db()
all_dfs = []


def calc_score(emb1, emb2):
    query = emb2.unsqueeze(1)
    score = torch.mm(emb1, query)
    return score.squeeze(1)


def get_embeddings(model, ds, loader, use_flip=False):
    embeddings = np.zeros((len(ds), EMBEDDING_DIM), dtype=np.float32)
    all_logits = np.zeros((len(ds), NUM_CLASSES), dtype=np.float32)

    files_dict = {}
    idx = 0
    for batch in tqdm.tqdm(loader, desc="Counting embeddings"):
        imgs = batch.float().cuda()
        if use_flip:
            for img_num in range(len(imgs)):
                imgs[img_num] = torchvision.transforms.functional.hflip(imgs[img_num])

        logits, features = model(imgs.half())

        features = features.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        embeddings[idx : idx + len(features)] = features
        all_logits[idx : idx + len(features)] = logits

        for _ in range(len(features)):
            files_dict[ds.imlist.loc[idx][0]] = idx
            idx = idx + 1

    return files_dict, embeddings, all_logits


def correct_scores_cluster(submit_pairs_df, neighbors, all_logits, files_dict):
    scores_arr = submit_pairs_df["score"].values
    new_scores = np.zeros_like(scores_arr)
    classifier_trusts = np.zeros_like(scores_arr)

    softmax_all_logits = all_logits.copy()
    logits_exp = np.exp(softmax_all_logits)
    softmax_all_logits = logits_exp / np.expand_dims(np.sum(logits_exp, axis=1), axis=1)

    for row in tqdm.tqdm(submit_pairs_df.itertuples(), desc="Correcting scores"):
        cluster_score = scores_arr[row.Index]

        indx1 = files_dict[row.img1]
        indx2 = files_dict[row.img2]

        indices1 = neighbors[indx1]
        indices2 = neighbors[indx2]
        cluster_logits1 = softmax_all_logits[indices1]
        cluster_logits2 = softmax_all_logits[indices2]
        cluster_class1 = cluster_logits1.argmax(axis=1)
        cluster_class2 = cluster_logits2.argmax(axis=1)

        class1_conf = cluster_logits1[:, cluster_class1].mean()
        class2_conf = cluster_logits2[:, cluster_class2].mean()

        cluster_is_12_conf = 1 - (
            cluster_logits2[:, cluster_class2] - cluster_logits2[:, cluster_class1] + 1e-5
        ) / (cluster_logits2[:, cluster_class2] + cluster_logits2[:, cluster_class1] + 1e-5)
        cluster_is_21_conf = 1 - (
            cluster_logits1[:, cluster_class1] - cluster_logits1[:, cluster_class2] + 1e-5
        ) / (cluster_logits1[:, cluster_class1] + cluster_logits1[:, cluster_class2] + 1e-5)

        cluster_avg_conf = (cluster_is_21_conf.mean() + cluster_is_12_conf.mean()) / 2

        trust_to_cluster_classier = min(class1_conf, class2_conf)
        cluster_corrected_score = (
            trust_to_cluster_classier * cluster_avg_conf
            + (1 - trust_to_cluster_classier) * cluster_score
        )
        new_scores[row.Index] = cluster_corrected_score
        classifier_trusts[row.Index] = trust_to_cluster_classier

    submit_pairs_df["corrected_score"] = new_scores
    submit_pairs_df["conf"] = classifier_trusts

    return submit_pairs_df


all_dfs = []

for model_num, params in enumerate(MODELS):
    print(f"Model {model_num+1}/{len(MODELS)} in process...")
    model = src.models.classifier.EffNetv2(
        class_num=NUM_CLASSES,
        features_dim=EMBEDDING_DIM,
        classify=True,
        mix_prec=True,
        model_name=params["version"],
    )
    model.load_state_dict(torch.load(params["model"]))

    model = model.eval()
    model.classifier = model.classifier.float()
    model.backbone = model.backbone.half()
    model = model.cuda()

    inf_transforms = src.utils.transforms.CarsTransforms(
        params["target_shape"], IMAGE_MEAN, IMAGE_STD, augs=False
    )
    inference_dataset = src.datasets.CarsDataset.CarsDatasetInference(
        DATASET_ROOT_DIR, transforms=inf_transforms
    )
    inf_loader = DataLoader(
        dataset=inference_dataset,
        batch_size=params["bs"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    files_dict, embeddings, logits = get_embeddings(
        model, inference_dataset, inf_loader, use_flip=params["flip"]
    )
    submit_pairs_df = pd.read_csv(
        os.path.join(DATASET_ROOT_DIR, "public_test", "submission_list.csv")
    )

    scores_arr = np.empty(len(submit_pairs_df), dtype=float)
    nembeddings = normalize(embeddings)

    # concat database
    if params["use_db"]:
        nembeddings = np.concatenate([nembeddings, db_embeddings], axis=0)
        logits = np.concatenate([logits, db_one_hot_labels], axis=0)

    tnembeddings = torch.from_numpy(nembeddings).float().cuda()
    neighbors = torch.empty((len(tnembeddings), params["neightbors_num"]), dtype=torch.long)

    for i in tqdm.tqdm(range(0, len(tnembeddings), NEIGHBORS_COUNT_BATCH), "Counting neighbors"):
        scores = torch.mm(tnembeddings, tnembeddings[i : i + NEIGHBORS_COUNT_BATCH].T)
        neighbors[i : i + NEIGHBORS_COUNT_BATCH] = torch.topk(
            scores, params["neightbors_num"], dim=0
        ).indices.T

    for row in tqdm.tqdm(submit_pairs_df.itertuples(), desc="Counting scores"):
        indices1 = neighbors[files_dict[row.img1]]
        indices2 = neighbors[files_dict[row.img2]]
        scores_arr[row.Index] = torch.mm(tnembeddings[indices1], tnembeddings[indices2].T).mean()

    del tnembeddings

    submit_pairs_df["score"] = scores_arr
    submit_pairs_df = correct_scores_cluster(submit_pairs_df, neighbors, logits, files_dict)

    all_dfs.append(submit_pairs_df)


# MERGE
res_submit_pairs_df = all_dfs[0].copy()

all_confs = np.empty((len(all_dfs), len(res_submit_pairs_df)))
all_scores = np.empty((len(all_dfs), len(res_submit_pairs_df)))

for df_idx, df in enumerate(all_dfs):
    all_confs[df_idx] = df["conf"].values
    all_scores[df_idx] = df["corrected_score"].values

model_weights = np.array([m["weight"] for m in MODELS])
model_weights = model_weights / model_weights.sum()
for model_num, model_weight in enumerate(model_weights):
    all_confs[model_num] *= model_weight

conf_exp = np.exp(5 * all_confs)
classifier_weights = conf_exp / conf_exp.sum(axis=0)

res_submit_pairs_df["score"] = 0
for cw, s in zip(classifier_weights, all_scores):
    res_submit_pairs_df["score"] += cw * s

res_submit_pairs_df["score"] = res_submit_pairs_df["score"].clip(0, 1)
res_submit_pairs_df[["id", "score"]].to_csv(
    os.path.join(SUBMITS_DIR, f"{SUBMIT_NAME}.csv"), index=False
)
