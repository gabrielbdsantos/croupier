# coding: utf-8
"""Provide I/O functionalities."""

import pathlib
from os import PathLike

import numpy as np
from numpy.typing import NDArray


def export_design(
    desing_matrix: NDArray[np.floating],
    path: PathLike,
    single_file: bool = True,
    file_prefix: str = "trajectory",
    fmt: str = "%.6f",
) -> None:
    """Export a set of trajectories to individual files.

    Parameters
    ----------
    trajectory : np.ndarray
        The set of trajectories to be exported. It must be a
        three-dimensional matrix in which the first dimension represents
        the trajectories. The second and third dimensions represent the
        number of points and number of parameters in each trajectory.
    path : path-like
        Path to the output file or folder.
    single_file: bool, optional
        Choose whether to output all trajectories to a single file or to
        write each trajectory to a different files in the specified
        folder.
    file_prefix : str, optional
        The string used as a prefix for the each trajectory. It is used
        only if `single_file` is False. Each trajectory is written on a
        separe file, which follows the following naming convention:
            {output folder}/{name prefix}_{trajectory number}.txt
    fmt : str, optional
        A single format, a sequence of formats, or a multi-format
        string, e.g. ‘Iteration %d – %10.5f’, in which case delimiter is
        ignored.

    Returns
    -------
    None

    Raises
    ------
    PermissionError
        The user does not have permission to write to path.
    ValueError
        Expected a three-dimensional design matrix.
    """
    # Put the output folder name in terms of absolute path.
    path = pathlib.Path(path)

    if not single_file:
        # If the output folder does not exist, try to create it. It may happen
        # that the user does not have permission to write to the output folder.
        # If this does happen, then throw a PermissionError.
        path.parent.mkdir(parents=True, exist_ok=False)

    # Total number of trajectories
    num_trajectories = desing_matrix.shape[0]

    # If the number of dimensions of the trajectory matrix is different from
    # three, throw an ValueError.
    if desing_matrix.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional trajectory matrix."
            f" Found a matrix with {desing_matrix.ndim}"
        )

    for i, trajectory in enumerate(desing_matrix):
        output = (
            path
            if single_file
            else (
                path / f"{file_prefix}_{i:0{num_trajectories // 10 + 1}}.txt"
            )
        )

        kws = (
            {"header": f"Trajectory {i}", "footer": "--"}
            if single_file
            else {}
        )

        # Write the trajectory to file.
        with open(output, "w+") as file:
            np.savetxt(file, trajectory, fmt=fmt, **kws)
