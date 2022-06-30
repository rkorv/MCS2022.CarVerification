import matplotlib.pyplot as plt
import numpy as np


def denormilize_img(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def plot_imgs(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_rows * 16, num_cols * 16)
    )

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if row_title is not None:
                ax.set(xlabel=row_title[col_idx])

    plt.tight_layout()


def plot_img(imgs, row_title=None, **imshow_kwargs):
    plot_imgs([imgs], row_title, **imshow_kwargs)
