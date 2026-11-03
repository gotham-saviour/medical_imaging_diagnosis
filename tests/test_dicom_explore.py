from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.data_pipeline import dicom_explore as de

matplotlib.use("Agg")


## --------------------------------------------------------------------
# TEST 1 --> load_sample(name)
## --------------------------------------------------------------------


def test_load_sample() -> None:
    ds = de.load_sample()
    assert ds.Modality == "MR"

    with pytest.raises(FileNotFoundError):
        de.load_sample("non_existent_file.dcm")


## --------------------------------------------------------------------
# TEST 2 --> to_hounsfield_or_stored(ds)
## --------------------------------------------------------------------


def test_to_hounsfield_or_stored() -> None:

    ds = de.load_sample()
    scaled = de.to_hounsfield_or_stored(ds)

    assert scaled.dtype == np.float32
    assert scaled.shape == (ds.Rows, ds.Columns)


## ---------------------------------------------------------------------
# TEST 3 --> get_window(ds,scaled_pixels)
## ---------------------------------------------------------------------


def test_get_window() -> None:

    # 3.a: Scalar Tags give floats
    ds = de.load_sample()
    scaled = de.to_hounsfield_or_stored(ds)
    center, width = de.get_window(ds, scaled)
    assert isinstance(center, float)
    assert isinstance(width, float)

    # 3.b: MultiValue Tags [40,400] & [80,2000] give (40.0,80.0)
    ds = de.load_sample()
    ds.WindowCenter = [40, 400]
    ds.WindowWidth = [80, 2000]

    center, width = de.get_window(ds, scaled)
    assert (center, width) == (40.0, 80.0)

    # 3.c: Tags Deleted and slope:2 intercept:-1000 gives (1272.0,4036.0)
    ds = de.load_sample()
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = -1000.0
    del ds.WindowCenter
    del ds.WindowWidth

    re_scaled = de.to_hounsfield_or_stored(ds)
    assert de.get_window(ds, re_scaled) == (1272.0, 4036.0)

    # 3.d: Constant Image Raises a Value Error
    constant_image = np.full((5, 5), fill_value=7.0)

    with pytest.raises(ValueError):
        de.get_window(ds, constant_image)


## ---------------------------------------------------------------------
# TEST 4: apply_window(scaled_pixels,center,width)
## ---------------------------------------------------------------------


def test_apply_window() -> None:

    # 4.a : [-20, 0, 50, 100, 200]; centre:50: width:100 gives ->[0, 0, 127, 255, 255] as uint8
    scaled_pixels = np.array([-20, 0, 50, 100, 200])
    center = 50.0
    width = 100.0
    window_scaled = de.apply_window(scaled_pixels, center, width)
    expected = np.array([0, 0, 127, 255, 255], dtype=np.uint8)

    # We chose Truncated Value of 127.5 which is 127 rather than rounding to 128
    assert (window_scaled).dtype == np.uint8
    np.testing.assert_array_equal(window_scaled, expected)

    ## 4.b : 0 width raises ValueError
    width = 0.0

    with pytest.raises(ValueError):
        de.apply_window(scaled_pixels, center, width)


## ---------------------------------------------------------------------
# TEST 5: save_fig(image,title,out_path)
## ---------------------------------------------------------------------


def test_save_fig(tmp_path: Path) -> None:
    plt.close("all")  ## Ensuring None of the Image is loaded
    dummy_image = np.zeros((10, 10), dtype=np.uint8)
    img_file = tmp_path / "subdir" / "test_img.png"

    de.save_fig(image=dummy_image, title="MR Dummy", out_path=img_file)

    assert img_file.exists()
    assert plt.get_fignums() == []


## ---------------------------------------------------------------------
# TEST 6: print_tags(ds,tags) with capsys
## ---------------------------------------------------------------------


def test_print_tags(capsys: pytest.CaptureFixture[str]) -> None:

    ds = de.load_sample()
    tags_to_check = ["PatientBirthDate", "InstitutionName", "RescaleSlope"]

    de.print_tags(ds, tags_to_check)
    captured = capsys.readouterr().out

    lines = {}

    for line in captured.strip().splitlines():
        lines[line.split(":")[0].strip()] = line.split(":")[1].strip()

    assert lines["PatientBirthDate"] == "[Empty String]"
    assert (
        lines["InstitutionName"] == "TOSHIBA str"
    )  ## we have Value and Value's data type
    assert lines["RescaleSlope"] == "Attribute Not Available"


## ---------------------------------------------------------------------
# TEST 7: main(out_dir)
## ---------------------------------------------------------------------


def test_main(tmp_path: Path) -> None:
    ## Passing a String Path

    str_dir = tmp_path / "run_str"
    de.main(out_dir=str(str_dir))
    assert (str_dir / "dicom_raw.png").exists()
    assert (str_dir / "dicom_windowed.png").exists()

    ## Passing a Path Object
    path_dir = tmp_path / "run_path"
    de.main(out_dir=Path(path_dir))
    assert (path_dir / "dicom_raw.png").exists()
    assert (path_dir / "dicom_windowed.png").exists()


## ---------------------------------------------------------------------
# TEST 8 -> Repo Root Guard
## ---------------------------------------------------------------------


def test_repo_root_guard() -> None:

    assert (de.REPO_ROOT / "pyproject.toml").exists()
