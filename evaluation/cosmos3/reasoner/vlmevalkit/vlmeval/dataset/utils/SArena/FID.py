import os

import clip
import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from vlmeval.smp.file import LMUDataRoot
from .base_metric import BaseMetric
from .inception import InceptionV3


class FIDCalculator(BaseMetric):
    """
    Calculate FID and FID-C Score
    """
    def __init__(self, model_name='InceptionV3'):  # Fixed E251
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if self.model_name == 'ViT-B/32':
            self.dims = 512
            root = os.path.join(LMUDataRoot(), 'aux_models')
            model, preprocess = clip.load('ViT-B/32', download_root=root)
        elif self.model_name == 'InceptionV3':
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx]).to(self.device)
            preprocess = TF.Compose([TF.ToTensor()])

        self.model = model.to(self.device)
        self.preprocess = preprocess

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps  # Fixed E128
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def get_activations(self, images):
        dataset = ImageDataset(images, self.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)
        pred_arr = np.empty((len(images), self.dims))
        start_idx = 0
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            with torch.no_grad():
                if self.model_name == 'ViT-B/32':
                    pred = self.model.encode_image(batch).cpu().numpy()
                elif self.model_name == 'InceptionV3':
                    pred = self.model(batch)[0]

                    # If model output is not scalar, apply global spatial average pooling.
                    # This happens if you choose a dimensionality not equal 2048.
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                start_idx = start_idx + pred.shape[0]

        return pred_arr

    def calculate_activation_statistics(self, images):
        # Fixed E117 (Over-indentation)
        act = self.get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def pil_images_to_tensor(self, images_list):
        """Convert a list of PIL Images to a torch.Tensor."""
        tensors_list = [self.preprocess(img) for img in images_list]
        return torch.stack(tensors_list).to(self.device)  # BxCxHxW format

    def calculate_score(self, batch, update=True):
        m1, s1 = self.calculate_activation_statistics(batch['gt_im'])
        m2, s2 = self.calculate_activation_statistics(batch['pred_im'])
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        if update:
            self.meter.update(fid_value, len(batch['gt_im']))
        return fid_value, []


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, processor=None):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.processor is not None:
            img = self.processor(img)
        return img
