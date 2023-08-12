# coding: utf-8
"""Provide I/O functionalities."""

import pathlib
from os import PathLike

import numpy as np
from numpy.typing import NDArray


def export_design_to_directory(
    design_matrix: NDArray[np.floating],
    output_dir: PathLike,
    file_prefix: str = "trajectory",
    fmt: str = "%.6f",
    overwrite: bool = False,
) -> None:
    """Export a set of trajectories to individual files in a directory.

    Parameters
    ----------
    design_matrix : np.ndarray
        The set of trajectories to be exported. It must be a
        three-dimensional matrix in which the first dimension represents
        the trajectories. The second and third dimensions represent the
        number of points and number of parameters in each trajectory.
    output_dir : path-like
        Path to the output directory.
    file_prefix : str, optional
        The string used as a prefix for the each trajectory. Each
        trajectory is written on a separe file, which follows the
        following naming convention:
            {output folder}/{name prefix}_{trajectory number}.txt
    fmt : str, optional
        A single format, a sequence of formats, or a multi-format
        string, e.g. ‘Iteration %d – %10.5f’, in which case delimiter is
        ignored.
    overwrite : bool, optional
        Overwrite existing directory. Use it with caution, as it may
        lead to incosistent results.

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
    path = pathlib.Path(output_dir)

    # If the output folder does not exist, try to create it. It may
    # happen that the user does not have permission to write to the
    # output folder. If this does happen, then throw a PermissionError.
    # If the folder exists, throw a FileExistsError.
    path.mkdir(parents=True, exist_ok=overwrite)

    # Total number of trajectories
    num_trajectories = design_matrix.shape[0]
    num_trajectories_digits = int(np.ceil(np.log10(num_trajectories)) + 1.0)

    # If the number of dimensions of the trajectory matrix is different from
    # three, throw an ValueError.
    if design_matrix.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional trajectory matrix."
            f" Found a matrix with {design_matrix.ndim}"
        )

    for i, trajectory in enumerate(design_matrix):
        # Set the output format.
        output = path / f"{file_prefix}_{i:0{num_trajectories_digits}}.txt"

        # Write the trajectory to file.
        np.savetxt(output, trajectory, fmt=fmt)


def export_design_to_file(
    desing_matrix: NDArray[np.floating],
    output_file: PathLike,
    fmt: str = "%.6f",
    overwrite: bool = False,
) -> None:
    """Export a set of trajectories to individual files in a directory."""
    # Put the output folder name in terms of absolute path.
    path = pathlib.Path(output_file)

    # If the output file does not exist, try to create it. It may
    # happen that the user does not have permission to write to the
    # output folder. If this does happen, then throw a PermissionError.
    # If the file exists, throw a FileExistsError.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=overwrite)

    # Total number of trajectories
    num_trajectories = desing_matrix.shape[0]
    num_trajectories_digits = int(np.ceil(np.log10(num_trajectories)) + 1.0)

    # If the number of dimensions of the trajectory matrix is different from
    # three, throw an ValueError.
    if desing_matrix.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional trajectory matrix."
            f" Found a matrix with {desing_matrix.ndim}"
        )

    with open(path, "w+") as file:
        for i, trajectory in enumerate(desing_matrix):
            # Write the trajectory to file.
            np.savetxt(
                file,
                trajectory,
                fmt=fmt,
                header=f"Trajectory {i:0{num_trajectories_digits}}",
                footer="--",
            )
