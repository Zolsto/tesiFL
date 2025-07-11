# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This module contains the Dirichlet partitioner class and its implementations.
The class is a modified version of the one from the Flower library (license above).
The code has been modified to work with Pytorch style datasets.
removing the requirements to use Hugging Face datasets.

Classes:
    - DirichletPartitioner: Partitioner based on Dirichlet distribution.

Functions:
    None

Constants:
    None

Exceptions:
    ValueError: Raised when the number of partitions is not greater than zero.
    ValueError: Raised when the alpha values are not strictly greater than zero.
    ValueError: Raised when the alpha values are not of the correct length.
    ValueError: Raised when the number of partitions is greater than the number of samples.
    ValueError: Raised when the min_partition_size is not satisfied after 10 attempts.
    ValueError: Raised when the alpha values are not of the correct size.

Author: Matteo Caligiuri
"""


# Define what to export
__all__ = ["DirichletPartitioner"]

import warnings
from typing import Optional, Union, List
import numpy.typing as npt

import numpy as np
from torch.utils.data import Dataset, Subset

from .partitioner import Partitioner

# Define custom data types
NDArrayFloat = npt.NDArray[np.float64]


class DirichletPartitioner(Partitioner):
    """
    Partitioner based on Dirichlet distribution.

    Implementation based on Bayesian Nonparametric Federated Learning of Neural Networks
    https://arxiv.org/abs/1905.12022.

    The algorithm sequentially divides the data with each label. The fractions of the
    data with each label is drawn from Dirichlet distribution and adjusted in case of
    balancing. The data is assigned. In case the `min_partition_size` is not satisfied
    the algorithm is run again (the fractions will change since it is a random process
    even though the alpha stays the same).

    The notion of balancing is explicitly introduced here (not mentioned in paper but
    implemented in the code). It is a mechanism that excludes the partition from
    assigning new samples to it if the current number of samples on that partition
    exceeds the average number that the partition would get in case of even data
    distribution. It is controlled by`self_balancing` parameter.

    Args:
        num_partitions (int): The total number of partitions that the data will be divided into.
        partition_by (int): index of the column in the dataset that will be used for
            partitioning.
        alpha (Union[int, float, List[float], NDArrayFloat]): Concentration parameter to
            the Dirichlet distribution.
        min_partition_size (int): The minimum number of samples that each partitions will
            have (the sampling process is repeated if any partition is too small).
        self_balancing (bool): Whether assign further samples to a partition after the
            number of samples exceeded the average number of samples per partition. (True
            in the original paper's code although not mentioned in paper itself).
        shuffle (bool): Whether to randomize the order of samples. Shuffling applied after
            the samples assignment to partitions.
        seed (int): Seed used for dataset shuffling. It has no effect if `shuffle` is False.

    Example:
        >>> from flwr_datasets import FederatedDataset
        >>> from flwr_datasets.partitioner import DirichletPartitioner
        >>>
        >>> partitioner = DirichletPartitioner(num_partitions=10, partition_by="label",
        >>>                                    alpha=0.5, min_partition_size=10,
        >>>                                    self_balancing=True)
        >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
        >>> partition = fds.load_partition(0)
        >>> print(partition[0])  # Print the first example
        {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x127B92170>,
        'label': 4}
        >>> partition_sizes = [
        >>>     len(fds.load_partition(partition_id)) for partition_id in range(10)
        >>> ]
        >>> print(sorted(partition_sizes))
        [2134, 2615, 3646, 6011, 6170, 6386, 6715, 7653, 8435, 10235]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: int,
        shift_partition_id: int,
        shift_indices: int,
        alpha: Union[int, float, list[float], NDArrayFloat],
        min_partition_size: int = 10,
        self_balancing: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self.shift_partition_id = shift_partition_id
        self.shift_indices = shift_indices
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._avg_num_of_samples_per_partition: Optional[float] = None
        self._unique_classes: Optional[Union[list[int], list[str]]] = None
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def create_partitions(self) -> None:
        """
        Create all partitions.

        Args:
            None

        Returns:
            None
        """
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()

    def load_partition(self, partition_id: int) -> Dataset:
        """
        Load a partition based on the partition index.

        Args:
            partition_id (int): the index that corresponds to the requested partition

        Returns:
            dataset_partition (Dataset): single dataset partition
        """

        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        print(f"Partitioon id {partition_id}")
        return Subset(self.dataset, self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _initialize_alpha(
        self, alpha: Union[int, float, list[float], NDArrayFloat]
    ) -> NDArrayFloat:
        """Convert alpha to the used format in the code a NDArrayFloat.

        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.

        Parameters
        ----------
            alpha : Union[int, float, List[float], NDArrayFloat]
                Concentration parameter to the Dirichlet distribution

        Returns
        -------
        alpha : NDArrayFloat
            Concentration parameter in a format ready to used in computation.
        """
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, list):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "If passing alpha as a List, it needs to be of length of equal to "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            # pylint: disable=R1720
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length "
                    "equal to num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length"
                        " equal to num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(
                f"Alpha values should be strictly greater than zero. "
                f"Instead it'd be converted to {alpha}"
            )
        return alpha

    def _get_unique_classes(self) -> List[int]:
        """
        Get the unique classes in the dataset.

        Args:
            None

        Returns:
            List[int]: The unique classes in the dataset.
        """

        if self.labels is not None and self._partition_by == 1:
            return self.labels
        elif hasattr(self.dataset, "targets"):
            return list(set(self.dataset.targets))
        else:
            # Cycle through the dataset to get all the
            # unique classes based on the partition_by index
            unique_classes = []
            for sample in self.dataset:
                if sample[self._partition_by] not in unique_classes:
                    unique_classes.append(sample[self._partition_by])
            return unique_classes

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self._get_unique_classes()
        assert self._unique_classes is not None
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_partition = (
            len(self.dataset) / self._num_partitions
        )

        # Change targets list data type to numpy
        if hasattr(self.dataset, "targets"):
            targets = np.array(self.dataset.targets)  # pylint: disable=E1101
        else:
            targets = np.array([sample[self._partition_by] for sample in self.dataset])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        sampling_try = 0
        while True:
            # Prepare data structure to store indices assigned to partition ids
            partition_id_to_indices: dict[int, list[int]] = {}
            for nid in range(self._num_partitions):
                partition_id_to_indices[nid] = []

            # Iterated over all unique labels (they are not necessarily of type int)
            for k in self._unique_classes:
                # Access all the indices associated with class k
                indices_representing_class_k = np.nonzero(targets == k)[0]
                # Determine division (the fractions) of the data representing class k
                # among the partitions
                class_k_division_proportions = self._rng.dirichlet(self._alpha)
                nid_to_proportion_of_k_samples = {}
                for nid in range(self._num_partitions):
                    nid_to_proportion_of_k_samples[nid] = class_k_division_proportions[
                        nid
                    ]
                # Balancing (not mentioned in the paper but implemented)
                # Do not assign additional samples to the partition if it already has
                # more than the average numbers of samples per partition. Note that it
                # might especially affect classes that are later in the order. This is
                # the reason for more sparse division that the alpha might suggest.
                if self._self_balancing:
                    assert self._avg_num_of_samples_per_partition is not None
                    for nid in nid_to_proportion_of_k_samples.copy():
                        if (
                            len(partition_id_to_indices[nid])
                            > self._avg_num_of_samples_per_partition
                        ):
                            nid_to_proportion_of_k_samples[nid] = 0

                    # Normalize the proportions such that they sum up to 1
                    sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                    for nid, prop in nid_to_proportion_of_k_samples.copy().items():
                        nid_to_proportion_of_k_samples[nid] = prop / sum_proportions

                # Determine the split indices
                cumsum_division_fractions = np.cumsum(
                    list(nid_to_proportion_of_k_samples.values())
                )
                cumsum_division_numbers = cumsum_division_fractions * len(
                    indices_representing_class_k
                )
                # [:-1] is because the np.split requires the division indices but the
                # last element represents the sum = total number of samples
                # The default implementation uses a floor we change it to round
                # to better balance the partition distribution that tended to
                # accumulate too many saples to the last apartition
                indices_on_which_split = np.round(cumsum_division_numbers).astype(int)[:-1]

                split_indices = np.split(
                    indices_representing_class_k, indices_on_which_split
                )

                # Append new indices (coming from class k) to the existing indices
                for nid, indices in partition_id_to_indices.items():
                    indices.extend(split_indices[nid].tolist())

            # Determine if the indices assignment meets the min_partition_size
            # If it does not mean the requirement repeat the Dirichlet sampling process
            # Otherwise break the while loop
            min_sample_size_on_client = min(
                len(indices) for indices in partition_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break
            sample_sizes = [
                len(indices) for indices in partition_id_to_indices.values()
            ]
            alpha_not_met = [
                self._alpha[i]
                for i, ss in enumerate(sample_sizes)
                if ss == min(sample_sizes)
            ]
            mssg_list_alphas = (
                (
                    "Generating partitions by sampling from a list of very wide range "
                    "of alpha values can be hard to achieve. Try reducing the range "
                    f"between maximum ({max(self._alpha)}) and minimum alpha "
                    f"({min(self._alpha)}) values or increasing all the values."
                )
                if len(self._alpha.flatten().tolist()) > 0
                else ""
            )
            warnings.warn(
                f"The specified min_partition_size ({self._min_partition_size}) was "
                f"not satisfied for alpha ({alpha_not_met}) after "
                f"{sampling_try} attempts at sampling from the Dirichlet "
                f"distribution. The probability sampling from the Dirichlet "
                f"distribution will be repeated. Note: This is not a desired "
                f"behavior. It is recommended to adjust the alpha or "
                f"min_partition_size instead. {mssg_list_alphas}",
                stacklevel=1,
            )
            if sampling_try == 10:
                raise ValueError(
                    "The max number of attempts (10) was reached. "
                    "Please update the values of alpha and try again."
                )
            sampling_try += 1

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)

        # HERE!! Translate indices
        if self.shift_indices != 0 and self.shift_partition_id != 0:
            shifted_partition_id_to_indices = {}
            for key in partition_id_to_indices:
                tmp_list = []
                for value in partition_id_to_indices[key]:
                    new_value = value + self.shift_indices
                    tmp_list.append(new_value)
                shifted_partition_id_to_indices[key + self.shift_partition_id] = tmp_list
            self._partition_id_to_indices = shifted_partition_id_to_indices
        else:
            self._partition_id_to_indices = partition_id_to_indices
            
        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > len(self.dataset):
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
