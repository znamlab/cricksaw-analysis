import re
from pathlib import Path
import flexiznam as flz


def create_dataset_from_log(
    cellfinder_log, parent_id, flexilims_session, conflicts="abort", upload=False
):
    params = parse_log(cellfinder_log)
    # find the dataset on flexilims
    ds = flz.Dataset.from_origin(
        origin_id=parent_id,
        flexilims_session=flexilims_session,
        dataset_type="cellfinder",
        conflicts=conflicts,
    )
    # get path from output_dir
    # the exact root might have changed, keep part after project name
    project = flz.lookup_project(flexilims_session.project_id)
    path_parts = Path(params["output_dir"]).parts
    try:
        ds_path = Path(*path_parts[path_parts.index(project) :])
        ds.path = ds_path
    except ValueError:
        print(f"CANNOT FIND THE PATH TO THE DATASET FROM {params['output_dir']}")

    # set attributes
    ds.created = params.pop("created")
    ds.extra_attributes = params

    if upload:
        ds.update_flexilims()

    return ds


def parse_log(cellfinder_log):
    """Parse cellfinder log file

    Args:
        cellfinder_log (str): Full path to the cellfinder .log file

    Returns:
        dict: Parameters used for cellfinder run
    """
    with open(cellfinder_log, "r") as fh:
        txt = fh.read()
    lines = txt.splitlines()
    # parse only finished cellfinder runs
    out = dict()
    assert "Finished" in lines[-1], "Cellfinder did not finish"
    lines.pop(0)  # remove title line
    assert not lines.pop(0), "Empty line expected"
    date = re.match(
        r"Analysis carried out: (\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", lines.pop(0)
    ).group(1)
    day, time = date.split("_")
    out["created"] = f"{day} {time.replace('-', ':')}"
    txt, out_dir = lines.pop(0).split(": ")
    assert txt == "Output directory"
    out["output_dir"] = out_dir
    txt, code_dir = lines.pop(0).split(": ")
    assert txt == "Current directory"
    out["code_dir"] = code_dir
    txt, version = lines.pop(0).split(": ")
    assert txt == "Version"
    out["version"] = version

    # find the section that contains the variables
    while True:
        line = lines.pop(0)
        if "**  VARIABLES  **" in line:
            break
    assert not lines.pop(0), "Empty line expected"

    while True:
        line = lines.pop(0)
        if not line:
            break
        parts = line.split(":")
        if len(parts) == 1:
            assert parts[0] == "Namespace"
            continue
        elif len(parts) > 2:
            # we have a windows file path somewhere, refuse the : inside the 2nd part
            parts = [parts[0], ":".join(parts[1:])]
        key, value = [p.strip() for p in parts]
        out[key] = value

    return out
