"""An example of how to create a set of quasi-optimized trajectories."""

import tempfile

import croupier


def run() -> None:
    """Generate a set of quasi-optimized trajectories."""
    NUM_PARAMS = 4
    NUM_LEVELS = 4
    NUM_INITIAL_TRAJECTORIES = 1000
    NUM_TRAJECTORIES = 10
    OUTPUT_DIR = tempfile.mkdtemp()
    SEED = 0

    initial_trajectories = croupier.morris.trajectory_design(
        NUM_PARAMS, NUM_LEVELS, NUM_INITIAL_TRAJECTORIES, seed=SEED
    )

    trajectories = croupier.morris.quasi_optimized_trajectories(
        initial_trajectories, NUM_TRAJECTORIES
    )

    print(f"Saving trajectories to {OUTPUT_DIR}")
    croupier.io.export_design_to_directory(
        trajectories, OUTPUT_DIR, overwrite=True
    )


if __name__ == "__main__":
    run()
