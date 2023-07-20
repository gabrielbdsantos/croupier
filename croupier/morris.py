# coding: utf-8
"""Define sampling strategies for the Elementary Effects Method."""

from typing import Generator

import numba
import numpy as np
from numpy.typing import NDArray


def _delta(num_levels: int) -> float:
    """Compute the increment (delta) based on the number of levels.

    Parameters
    ----------
    num_levels : int
        The desired number of equally spaced increment levels.

    Returns
    -------
    float
        The increment level.

    Raises
    ------
    ValueError
        The number of increment levels should be greater than zero.
    RuntimeError
        The number of increment levels is odd. It has been already
        proven that an even number is better, providing an uniform
        distribution of the initial base values.
    """
    if num_levels <= 1:
        raise ValueError(
            "The number of increment levels should be greater than one."
        )

    if num_levels % 2 != 0:
        raise RuntimeError(
            "Although an odd number of levels could be"
            " accepted, an even number is proven to be a"
            " better choice."
        )

    return num_levels / (2.0 * (num_levels - 1))


@numba.njit(parallel=True)
def _euclidian_distance_matrix(trajectories: np.ndarray) -> np.ndarray:
    """Compute the euclidian distance between any two trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        The array of trajectories.

    Returns
    -------
    np.ndarray
        The distance matrix, which stores the euclidian distance in the upper
        triangular matrix and the squared euclidian distance in the lower
        triangular matrix. The indices (i, j) represent the distance between
        trajectories i and j.
    """
    # Get the total number of trajectories and the number of parameters.
    num_trajectories = trajectories.shape[0]

    # Initialize the distance matrix with zeros.
    d = np.zeros((num_trajectories, num_trajectories))

    # Get all possible combinations of trajectories.
    m_indices, l_indices = np.triu_indices_from(d, k=1)

    # Loop through all combinations
    for i in numba.prange(m_indices.shape[0]):
        _m, _l = m_indices[i], l_indices[i]

        # Compute the euclidian distance between trajectories _m and _l, and
        # store the result in the upper triangular matrix.
        for m in trajectories[_m]:
            for l in trajectories[_l]:
                d[_m, _l] += np.sqrt(np.square(m - l).sum())

        # Square the euclidian distance between _m and _l, and store the result
        # in the lower triangular matrix.
        d[_l, _m] = d[_m, _l] * d[_m, _l]

    # Return the distance matrix containing both the euclidian distance in the
    # upper triangular matrix and the squared euclidian distance in the lower
    # triangular matrix. Even though the method proposed by Campolong et. al.
    # only uses the squared euclidian distance from now on, the function
    # returns both values, envisioning future usage.
    return d


@numba.njit(parallel=True)
def _subset_distance(
    distance_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute all subsets distances to the original set of distances.

    The algorithm follows the method proposed in [1].

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix or the original set.

    Returns
    -------
    np.ndarray
        A one-dimensional array of the distance of all subsets to the
        original matrix.

    References
    ----------
    [1] Q. Ge and M. Menendez, “An Efficient Sensitivity Analysis
        Approach for Computationally Expensive Microscopic Traffic
        Simulation Models,” IJT, vol. 2, no. 2, pp. 49–64, Aug. 2014,
        doi: 10.14257/ijt.2014.2.2.04.
    """
    # Get the number of possible subsets.
    num_subsets = distance_matrix.shape[0]

    # Initialize the subset distance matrix with zeros.
    Ds_p = np.zeros(num_subsets)

    # Compute the influence
    for i in numba.prange(num_subsets):
        # Initiate a null matrix for using as maked indices.
        masked_indices = np.zeros((num_subsets, num_subsets))

        # Select the row and column corresponding to i, and set
        # the value to one.
        for j in numba.prange(i):
            masked_indices[i, j] = 1

        for j in numba.prange(i + 1, num_subsets):
            masked_indices[j, i] = 1

        # Compute the subset distance.
        Ds_p[i] = (distance_matrix * masked_indices).sum()

    return Ds_p


@numba.njit(parallel=False)
def quasi_optimized_trajectories(
    trajectories: NDArray[np.floating], num_qot: int
) -> NDArray[np.floating]:
    """Find the set of quasi-optimal trajectories.

    Parameters
    ----------
    trajectory : np.ndarray
        A set of randomly generated trajectories.
    num_qot : int
        The number of quasi-optimized trajectories to be selected out of
        the original set of randomly generated trajectories.

    Returns
    -------
    np.ndarray
        The set of quasi-optimized trajectories.
    """
    # The number of randomly generated trajectories.
    num_trajectories = trajectories.shape[0]

    # The necessary number of iterations.
    num_iterations = num_trajectories - num_qot
    if num_iterations <= 0:
        raise ValueError(
            "The given set of trajectories is smaller than the"
            " desired number of quasi-optimized trajectories."
        )

    # Index of the trajectory that is to be deleted.
    to_delete = 0

    # Compute the distance matrix and the subset
    Ds = _euclidian_distance_matrix(trajectories)
    Ds_p = _subset_distance(Ds)

    # The following for-loop cannot be executed in parallel.
    for _ in numba.prange(num_iterations):
        # Select the trajectory that contributes the less to the total
        # distance, and mark it to be deleted.
        to_delete = np.where(Ds_p == Ds_p.min())[0][0]

        # Deduct the contribution of the least influential trajectory on the
        # subset distance of other trajectories.
        for j in numba.prange(to_delete):
            Ds_p[j] = Ds_p[j] - Ds[to_delete, j]

        for j in numba.prange(to_delete + 1, Ds_p.shape[0]):
            Ds_p[j] = Ds_p[j] - Ds[j, to_delete]

        # Remove the least influential trajectory from the distance matrix
        Ds = np.vstack((Ds[:to_delete], Ds[to_delete + 1 :]))
        Ds = np.hstack((Ds[:, :to_delete], Ds[:, to_delete + 1 :]))

        # Remove the least influential trajectory from the subset distance matrix.
        Ds_p = np.hstack((Ds_p[:to_delete], Ds_p[to_delete + 1 :]))

        # Exclude the least influential trajectory from the list of trajectories.
        trajectories = np.vstack(
            (trajectories[:to_delete], trajectories[to_delete + 1 :])
        )

    return trajectories


def trajectory_design(
    num_params: int,
    num_levels: int,
    num_trajectories: int = 1,
    unique: bool = True,
    seed: int = -1,
    max_retries: int = 10000,
) -> NDArray[np.floating]:
    """Generate a design matrix following the trajectory strategy.

    Parameters
    ----------
    num_params : int
        The number of parameters (k) defining the problem.
    num_levels : int
        The number of equally spaced increment levels (p).
    num_trajectories : int, optional
        The number of trajectories (N) to be generated.
    unique : bool, optional
        Only return unique trajectories.
    seed : int, optional
        Reseed a legacy MT19937 BitGenerator.
    max_retries: int, optional
        The maximum number of retries when generating a set of unique
        trajectories.

    Returns
    -------
    np.ndarray
        The design matrix.

    Notes
    -----
    The implementation is based on the comments found in the 3rd Chapter
    of [1].

    [1] Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni,
        J., Gatelli, D., Saisana, M., & Tarantola, S. (2008). Global
        sensitivity analysis: The primer. John Wiley.
    """
    # If necessary, set the seed for NumPy.
    if seed >= 0:
        np.random.seed(seed)

    def unique_concat(
        x: NDArray[np.floating], y: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return np.unique(np.concatenate((x, y)), axis=0)

    def only_concat(
        x: NDArray[np.floating], y: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return np.concatenate((x, y))

    combine_fn = unique_concat if unique else only_concat

    # Compute the increment delta based on the number of levels (p).
    delta = _delta(num_levels)

    # A (k+1)-by-k matrix of zeros and ones, in which there are only two rows
    # that differ in the jth element. A convenient choice for B is a strictly
    # lower triangular matrix of ones.
    B = np.tril(np.ones((num_params + 1, num_params)), -1)

    # A simple (k+1)-by-k matrix of ones.
    J = np.ones((num_params + 1, num_params))

    # Generate a dummy trajectory.
    B_star = np.zeros((1, num_params + 1, num_params)) - 1

    # Start the attempt counter.
    attempt = 0

    while B_star.shape[0] < (num_trajectories + 1) and attempt < (
        num_trajectories + max_retries
    ):
        # Increment the attempt counter.
        attempt += 1

        # Generate a randomly chosen base value.
        x_star = np.random.choice(
            np.linspace(0, 1 - delta, int(num_levels / 2)), num_params
        )

        # A k-dimensional diagonal matrix in which each element is either +1 or
        # -1 with equal probability. It states whether the factors will
        # increase or decrease along the trajectory.
        D_star = np.diag(np.random.choice([-1, 1], num_params))

        # A k-by-k random permutation matrix in which each row contains one
        # element equal to 1, all others are 0, and no two rows have ones in
        # the same columns. The matrix gives the order in which factors are
        # moved by shuffling the values of an identity matrix of size k-by-k.
        np.random.shuffle(P_star := np.eye(num_params, num_params))

        # Compute the random permutation matrix.
        _b = (delta / 2) * ((2 * B - J) @ D_star + J)

        # Append the trajectory to the design matrix.
        B_star = combine_fn(B_star, ((J * x_star + _b) @ P_star)[None, :, :])

    # Skip the first (dummy) trajectory.
    return B_star[1:]


def radial_design(
    base_points: NDArray[np.floating],
    incremental_points: Generator[NDArray[np.floating], None, None],
    min_distance: float = 0.0,
):
    """Define a generic strategy for radial-like trajectory designs."""
    # Extract the number of trajectories and the number of parameters from the
    # matrix of base points.
    num_trajectories, num_params = base_points.shape

    # Helper function to evaluate the distance between two points. If a minimal
    # distance is defined, it will only return incremental points that
    # introduce a perturbation greater than the given minimal distance.
    # Otherwise, if no minimal distance is defined, the helper function always
    # return the next incremental points in line.
    def increment(index: int) -> NDArray[np.floating]:
        while np.any(
            np.isclose(
                base_points[index],
                inc := next(incremental_points),
                atol=min_distance,
            )
        ):
            pass

        return inc

    J = np.ones((num_params + 1, num_params))
    B = np.eye(num_params + 1, num_params, k=-1)

    return np.asfarray(
        [
            base_points[x] * (J - B) + increment(x) * B
            for x in range(num_trajectories)
        ]
    )
