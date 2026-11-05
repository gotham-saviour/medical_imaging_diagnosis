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
    tags_to_check = [
        "PatientID",
        "PatientBirthDate",
        "InstitutionName",
        "RescaleSlope",
        "StudyTime",
        "Manufacturer",
        "DeviceSerialNumber",
        "ImagePositionPatient",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    ]

    de.print_tags(ds, tags_to_check)
    captured = capsys.readouterr().out

    lines = {}

    for line in captured.strip().splitlines():
        lines[line.split(":")[0].strip()] = line.split(":")[1].strip()

    assert lines["StudyInstanceUID"] == "1.3.6.1.4.1.5962.1.2.4.20040826185059.5457 UID"
    assert (
        lines["SeriesInstanceUID"] == "1.3.6.1.4.1.5962.1.3.4.1.20040826185059.5457 UID"
    )
    assert (
        lines["SOPInstanceUID"] == "1.3.6.1.4.1.5962.1.1.4.1.1.20040826185059.5457 UID"
    )
    assert lines["PatientID"] == "4MR1 str"
    assert lines["PatientBirthDate"] == "[Empty String]"
    assert (
        lines["InstitutionName"] == "TOSHIBA str"
    )  ## we have Value and Value's data type
    assert lines["RescaleSlope"] == "Attribute Not Available"
    assert lines["StudyTime"] == "185059 str"
    assert lines["Manufacturer"] == "TOSHIBA_MEC str"
    assert lines["DeviceSerialNumber"] == "-0000200 str"
    assert lines["ImagePositionPatient"] == "[-83.9063, -91.2000, 6.6406] MultiValue"


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


## ---------------------------------------------------------------------
# TEST 9 -> distinct_pixel_values(image)
## ---------------------------------------------------------------------


def test_distinct_pixel_values() -> None:

    dummy_array = np.array([1, 4, 5, 5])
    unique_values = de.distinct_pixel_values(dummy_array)
    de.load_sample()
    assert unique_values == 3


## ---------------------------------------------------------------------
# TEST 10 -> scaled_quantization(image)
## ---------------------------------------------------------------------


def test_scaled_quantization() -> None:

    ds = de.load_sample()
    raw_pixels = ds.pixel_array
    quantized_8bit = de.scaled_quantization(raw_pixels)
    unique_pixels = np.unique_values(quantized_8bit)

    assert len(unique_pixels) == 225


## ---------------------------------------------------------------------
# TEST 11 -> count_outside_window()
## ---------------------------------------------------------------------


def test_count_outside_window() -> None:

    dummy_image = np.arange(-250, 250).reshape(25, 20)
    values_count_outside_window = de.count_outside_window(dummy_image, 0.0, 400)
    assert values_count_outside_window == (50, 49)


## ---------------------------------------------------------------------
# TEST 12 -> Sample Intensity Claims Verification
## ---------------------------------------------------------------------


def test_sample_intensity_report() -> None:
    # 12.a: Verifying number of unique pixel values in Raw Image
    ds = de.load_sample()

    unique_values = de.distinct_pixel_values(ds.pixel_array)
    assert unique_values == 1128

    # 12.b: Verifying Unique Pixel Values after windowing in Sample
    ds = de.load_sample()
    scaled_pixels = de.to_hounsfield_or_stored(ds)
    window_center, window_width = de.get_window(ds, scaled_pixels)
    windowed_pixels = de.apply_window(scaled_pixels, window_center, window_width)
    unique_values_windowed = de.distinct_pixel_values(windowed_pixels)
    assert unique_values_windowed == 204

    # 12.c:Verifying  Number of Pixels Clipped during Window Scaling of Sample
    ds = de.load_sample()
    scaled_pixels = de.to_hounsfield_or_stored(ds)
    window_center, window_width = de.get_window(ds, scaled_pixels)
    pixels_clipped = sum(
        de.count_outside_window(scaled_pixels, window_center, window_width)
    )

    assert pixels_clipped == 222

    # 12.d : Verifying unique pixel count when Sample is 8-bit quantized with no Clipping
    ds = de.load_sample()
    scaled_pixels = de.to_hounsfield_or_stored(ds)
    quantized_8_bit_image = de.scaled_quantization(scaled_pixels)
    unique_pixels_quantized = de.distinct_pixel_values(quantized_8_bit_image)

    assert unique_pixels_quantized == 225
