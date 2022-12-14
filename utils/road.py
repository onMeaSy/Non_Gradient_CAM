# Implementations of our imputation models.
import torch
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Callable
import cv2


class PerturbationConfidenceMetric:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model(perturbated_tensors)
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)

        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation

        if return_visualization:
            return result, perturbated_tensors
        else:
            return result


class RemoveMostRelevantFirst:
    def __init__(self, percentile, imputer):
        self.percentile = percentile
        self.imputer = imputer

    def __call__(self, input_tensor, mask):
        imputer = self.imputer
        if self.percentile != 'auto':
            threshold = np.percentile(mask.cpu().numpy(), self.percentile)
            binary_mask = np.float32(mask < threshold)
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(mask * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_mask = torch.from_numpy(binary_mask)
        binary_mask = binary_mask.to(mask.device)
        return imputer(input_tensor, binary_mask)


class RemoveLeastRelevantFirst(RemoveMostRelevantFirst):
    def __init__(self, percentile, imputer):
        super(RemoveLeastRelevantFirst, self).__init__(percentile, imputer)

    def __call__(self, input_tensor, mask):
        return super(RemoveLeastRelevantFirst, self).__call__(
            input_tensor, 1 - mask)


class AveragerAcrossThresholds:
    def __init__(
        self,
        imputer,
        percentiles=[
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90]):
        self.imputer = imputer
        self.percentiles = percentiles

    def __call__(self,
                 input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module):
        scores = []
        for percentile in self.percentiles:
            imputer = self.imputer(percentile)
            scores.append(imputer(input_tensor, cams, targets, model))
        return np.mean(np.float32(scores), axis=0)


# The weights of the surrounding pixels
neighbors_weights = [((1, 1), 1 / 12),
                     ((0, 1), 1 / 6),
                     ((-1, 1), 1 / 12),
                     ((1, -1), 1 / 12),
                     ((0, -1), 1 / 6),
                     ((-1, -1), 1 / 12),
                     ((1, 0), 1 / 6),
                     ((-1, 0), 1 / 6)]


class NoisyLinearImputer:
    def __init__(self,
                 noise: float = 0.01,
                 weighting: List[float] = neighbors_weights):
        """
                Noisy linear imputation.
                noise: magnitude of noise to add (absolute, set to 0 for no noise)
                weighting: Weights of the neighboring pixels in the computation.
                List of tuples of (offset, weight)
        """
        self.noise = noise
        self.weighting = neighbors_weights

    @staticmethod
    def add_offset_to_indices(indices, offset, mask_shape):
        """ Add the corresponding offset to the indices.
    Return new indices plus a valid bit-vector. """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]
        valid = ((cord0 < 0) | (cord1 < 0) |
                 (cord0 >= mask_shape[0]) |
                 (cord1 >= mask_shape[1]))
        return ~valid, indices + offset[0] * mask_shape[1] + offset[1]

    @staticmethod
    def setup_sparse_system(mask, img, neighbors_weights):
        """ Vectorized version to set up the equation system.
                mask: (H, W)-tensor of missing pixels.
                Image: (H, W, C)-tensor of all values.
                Return (N,N)-System matrix, (N,C)-Right hand side for each of the C channels.
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
    # Indices that are imputed in the flattened mask:
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))
        numEquations = len(indices)
    # System matrix:
        A = lil_matrix((numEquations, numEquations))
        b = np.zeros((numEquations, img.shape[0]))
    # Sum of weights assigned:
        sum_neighbors = np.ones(numEquations)
        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            # Take out outliers
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(
                indices, offset, mask.shape)
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()
            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T
            # Add weights to the system (left hand side)
# Find coordinates in the system.
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5]
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
            A[has_no_values_ids, variable_ids] = weight
            # Reduce weight for invalid
            sum_neighbors[np.argwhere(valid == 0).flatten()] = \
                sum_neighbors[np.argwhere(valid == 0).flatten()] - weight

        A[np.arange(numEquations), np.arange(numEquations)] = -sum_neighbors
        return A, b

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        """ Our linear inputation scheme. """
        """
		This is the function to do the linear infilling
		img: original image (C,H,W)-tensor;
		mask: mask; (H,W)-tensor
		"""
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
    # Indices that need to be imputed.
        indices_linear = np.argwhere(maskflt == 0).flatten()
        # Set up sparse equation system, solve system.
        A, b = NoisyLinearImputer.setup_sparse_system(
            mask.numpy(), img.numpy(), neighbors_weights)
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        # Fill the values with the solution of the system.
        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise * \
            torch.randn_like(res.t())

        return img_infill.reshape_as(img)


class ROADMostRelevantFirst(PerturbationConfidenceMetric):
    def __init__(self, percentile=80):
        super(ROADMostRelevantFirst, self).__init__(
            RemoveMostRelevantFirst(percentile, NoisyLinearImputer()))


class ROADLeastRelevantFirst(PerturbationConfidenceMetric):
    def __init__(self, percentile=20):
        super(ROADLeastRelevantFirst, self).__init__(
            RemoveLeastRelevantFirst(percentile, NoisyLinearImputer()))


class ROADMostRelevantFirstAverage(AveragerAcrossThresholds):
    def __init__(self, percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
        super(ROADMostRelevantFirstAverage, self).__init__(
            ROADMostRelevantFirst, percentiles)


class ROADLeastRelevantFirstAverage(AveragerAcrossThresholds):
    def __init__(self, percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
        super(ROADLeastRelevantFirstAverage, self).__init__(
            ROADLeastRelevantFirst, percentiles)


class ROADCombined:
    def __init__(self, percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
        self.percentiles = percentiles
        self.morf_averager = ROADMostRelevantFirstAverage(percentiles)
        self.lerf_averager = ROADLeastRelevantFirstAverage(percentiles)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module):

        scores_lerf = self.lerf_averager(input_tensor, cams, targets, model)
        scores_morf = self.morf_averager(input_tensor, cams, targets, model)
        return (scores_lerf - scores_morf) / 2