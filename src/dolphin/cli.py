import os


def _set_default_thread_limits():
    """Set conservative thread limits before any numerical libraries import.

    These are defaults; they can be overridden by the worker settings in config.
    This prevents libraries from spawning unlimited threads on import.
    """
    # Only set if not already set by user
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

    # JAX/XLA defaults - critical for preventing massive thread spawning
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


# Set thread limits BEFORE any imports
_set_default_thread_limits()


def main() -> int:
    """Top-level command line interface to the workflows."""
    import sys

    from dolphin import __version__

    # TODO: Is there any way to slot this into tyro's argparse
    # Only found this hacky way
    # https://github.com/brentyi/tyro/issues/132#issuecomment-1978319762
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        print(__version__)
        raise SystemExit(os.EX_OK)

    import tyro

    from dolphin.filtering import filter_rasters
    from dolphin.timeseries import run as run_timeseries
    from dolphin.unwrap import run as run_unwrap
    from dolphin.workflows._cli_config import ConfigCli

    tyro.extras.subcommand_cli_from_dict(
        {
            "run": run_cli,
            "config": ConfigCli,
            "unwrap": run_unwrap,
            "timeseries": run_timeseries,
            "filter": filter_rasters,
        },
        prog=__package__,
    )

    return os.EX_OK


def run_cli(
    config_file: str,
    /,
    debug: bool = False,
) -> None:
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    from .workflows import displacement
    from .workflows.config import DisplacementWorkflow

    cfg = DisplacementWorkflow.from_yaml(config_file)
    displacement.run(cfg, debug=debug)
