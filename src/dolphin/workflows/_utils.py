from __future__ import annotations

from pathlib import Path

from dolphin._log import get_log

from .config import Workflow

logger = get_log(__name__)


def _create_burst_cfg(
    cfg: Workflow,
    burst_id: str,
    grouped_slc_files: dict[str, list[Path]],
    grouped_amp_mean_files: dict[str, list[Path]],
    grouped_amp_dispersion_files: dict[str, list[Path]],
) -> Workflow:
    cfg_temp_dict = cfg.model_dump(exclude={"cslc_file_list"})

    # Just update the inputs and the work directory
    top_level_work = cfg_temp_dict["work_directory"]
    cfg_temp_dict.update({"work_directory": top_level_work / burst_id})
    cfg_temp_dict["cslc_file_list"] = grouped_slc_files[burst_id]
    cfg_temp_dict["amplitude_mean_files"] = grouped_amp_mean_files[burst_id]
    cfg_temp_dict["amplitude_dispersion_files"] = grouped_amp_dispersion_files[burst_id]
    return Workflow(**cfg_temp_dict)


def _remove_dir_if_empty(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass
