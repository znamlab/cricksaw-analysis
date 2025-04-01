import re
import warnings
from pathlib import Path

import flexiznam as flz
import yaml


def cricksaw_dataset_from_metadata(
    cricksaw_data, parent_id, flexilims_session, conflicts="abort", upload=False
):
    """Create a cricksaw dataset from a cricksaw recipe file

    Args:
        cricksaw_data (str): Path to cricksaw folder or recipe file
        parent_id (str): Hexadecimal id of the parent dataset on flexilims
        flexilims_session (flexilims.Session): Flexilims session object, must have valid
            project_id
        conflicts (str, optional): What to do if a cellfinder dataset already exists for
            this parent. See `flz.Dataset.from_origin` for more. Defaults to "abort".
        upload (bool, optional): Upload dataset to flexilims. Defaults to False.

    Returns:
        flexiznam.Dataset: Dataset object
    """
    metadata, recipe = get_cricksaw_metadata(cricksaw_data)

    start_time = metadata["Acquisition"]["acqStartTime"].replace("/", "-")
    ds = flz.Dataset.from_origin(
        origin_id=parent_id,
        flexilims_session=flexilims_session,
        conflicts=conflicts,
        dataset_type="cricksaw",
    )

    ds.created = start_time
    ds.extra_attributes = metadata

    if Path(flz.PARAMETERS["data_root"]["raw"]) in recipe.parents:
        ds.is_raw = True
        ds.path = recipe.relative_to(flz.PARAMETERS["data_root"]["raw"])
    elif Path(flz.PARAMETERS["data_root"]["processed"]) in recipe.parents:
        ds.is_raw = False
        ds.path = recipe.relative_to(flz.PARAMETERS["data_root"]["processed"])
    else:
        print(
            "CANNOT FIND THE PATH TO THE DATASET. "
            + f"{recipe} does not contain raw nor processed root."
        )
        print("Dataset won't be uploaded. Change path manually first.")
        return ds

    if upload:
        # remove warnings about wrong case. Flexilims arguments are always displayed
        # lowercase but we don't care here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.update_flexilims()

    return ds


def get_cricksaw_metadata(cricksaw_data):
    """Read cricksaw recipe file

    Args:
        cricksaw_data (str): Path to cricksaw folder or recipe file

    Returns:
        dict: Dictionary of metadata information
        pathlib.Path: path to recipe file
    """
    cricksaw_data = Path(cricksaw_data)
    assert cricksaw_data.exists(), f"Folder {cricksaw_data} does not exist"
    if cricksaw_data.is_dir():
        recipe_file = list(cricksaw_data.glob("recipe*.yml"))
        if len(recipe_file) != 1:
            raise IOError("Found %d recipe files" % len(recipe_file))
        recipe_file = recipe_file[0]
    else:
        assert cricksaw_data.suffix == ".yml", "Not a folder nor a recipe file"
        recipe_file = cricksaw_data

    with open(recipe_file) as f:
        recipe = yaml.safe_load(f)
    return recipe, recipe_file


def cellfinder_dataset_from_log(
    cellfinder_log, parent_id, flexilims_session, conflicts="abort", upload=False
):
    """Create a cellfinder dataset from a cellfinder log file

    Args:
        cellfinder_log (str): Path to cellfinder log file
        parent_id (str): Hexadecimal id of the parent dataset on flexilims
        flexilims_session (flexilims.Session): Flexilims session object, must have valid
            project_id
        conflicts (str, optional): What to do if a cellfinder dataset already exists for
            this parent. See `flz.Dataset.from_origin` for more. Defaults to "abort".
        upload (bool, optional): Upload dataset to flexilims. Defaults to False.

    Returns:
        flexiznam.Dataset: Dataset object
    """
    params = parse_cellfinder_log(cellfinder_log)
    # find the dataset on flexilims
    ds = flz.Dataset.from_origin(
        origin_id=parent_id,
        flexilims_session=flexilims_session,
        dataset_type="cellfinder",
        conflicts=conflicts,
    )
    # set attributes
    ds.created = params.pop("created")
    ds.extra_attributes = params

    # get path from output_dir
    # the exact root might have changed, keep part after project name
    project = flz.lookup_project(flexilims_session.project_id)
    path_parts = Path(params["output_dir"]).parts
    try:
        ds_path = Path(*path_parts[path_parts.index(project) :])
        ds.path = ds_path
    except ValueError:
        print(f"CANNOT FIND THE PATH TO THE DATASET FROM {params['output_dir']}")
        print("Dataset won't be uploaded. Change path manually first.")
        return ds

    if upload:
        ds.update_flexilims()

    return ds


def parse_cellfinder_log(cellfinder_log):
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
