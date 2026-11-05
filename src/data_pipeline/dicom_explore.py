from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset

## Root Directory of the Project
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sample(name: str = "MR_small.dcm") -> Dataset:
    """Load sample DICOM MRI Scan File from pre-built pydicom datasets"""

    ds = get_testdata_file(name, read=True)

    if ds is None:
        raise FileNotFoundError(f"No such file exists: {name}")

    else:
        return ds


def print_tags(ds: Dataset, tags: list[str]) -> None:
    """prints the value of provided DICOM file Attributes if present"""

    for tag in tags:
        if hasattr(ds, tag):
            val = getattr(ds, tag)
            if str(val).strip() == "":
                print(f"{tag:<27}: [Empty String]")
            else:
                print(f"{tag:<27}: {val} {type(val).__name__}")
        else:
            print(f"{tag:<27}: Attribute Not Available")


def to_hounsfield_or_stored(ds: Dataset) -> np.ndarray:
    """Rescales Raw Pixel values to Hounsfield Units(H.U.)"""
    rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    raw_pixels = ds.pixel_array
    hu_pixels = (raw_pixels * rescale_slope + rescale_intercept).astype(np.float32)

    return hu_pixels


def get_window(ds: Dataset, scaled_pixels: np.ndarray) -> tuple[float, float]:
    """Retrieving the Window Centre And Window Width Values from the Dataset"""
    try:
        if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
            window_center = float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth[0])

        else:
            window_center = float(ds.WindowCenter)
            window_width = float(ds.WindowWidth)

    except AttributeError:
        print(
            "DICOM file does not contain Windowing tags\n"
            "Using Min-Max Intensity Scaling Using Scaled Pixels"
        )
        # If missing, it defaults to full intensity range of the pixel array
        p_max = scaled_pixels.max()
        p_min = scaled_pixels.min()

        window_center = (p_max + p_min) / 2.0
        window_width = p_max - p_min

    if window_width <= 0.0:
        raise ValueError(f"Window Width must be Positive, got {window_width}")
    else:
        return (float(window_center), float(window_width))


def apply_window(scaled_pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    """Applies the Window Scaling to the Scaled Hounsfield Pixels"""

    if width <= 0.0:
        raise ValueError(f"Window Width must be Positive, got {width}")

    else:
        upper_bound = center + (width / 2)
        lower_bound = center - (width / 2)

        window_pixels = np.clip(scaled_pixels, lower_bound, upper_bound)

        window_pixels = (
            ((window_pixels - lower_bound) / (upper_bound - lower_bound)) * 255
        ).astype(np.uint8)

        return window_pixels


def save_fig(image: np.ndarray, title: str, out_path: Path) -> None:
    """Saving the Figure in the Provided Output path"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def distinct_pixel_values(image: np.ndarray) -> int:
    """Helper function to retrieve the Number of Distinct Pixel Values in an Image"""
    unique_values = np.unique_values(image)
    return len(unique_values)


def scaled_quantization(image: np.ndarray) -> np.ndarray:
    """Helper Function Converts Image from Full range to 8 bit with no Clipping"""
    p_max = image.max()
    p_min = image.min()

    if p_max == p_min:
        raise ValueError("The Minimum and Maximum Intensities cannt be equal")

    full_range_normalized = (image - p_min) / (p_max - p_min)

    full_range_8_bit = (full_range_normalized * 255).astype(np.uint8)

    return full_range_8_bit


def count_outside_window(
    image: np.ndarray, center: float, width: float
) -> tuple[int, int]:
    """Getting the Count of Pixels Values which are clipped during Window Clipping"""

    if width <= 0.0:
        raise ValueError(f"Window Width must be Positive, got {width}")

    else:
        upper_bound = center + (width / 2)
        lower_bound = center - (width / 2)

        values_above_upper_bound = int((image > upper_bound).sum())
        values_below_lower_bound = int((image < lower_bound).sum())

        return (values_below_lower_bound, values_above_upper_bound)


def main(out_dir: str | Path | None = None) -> None:

    ## Check if the Output path is Available

    if not out_dir:
        out_dir = Path("docs/images")
    else:
        out_dir = Path(out_dir)

    out_path = REPO_ROOT / out_dir

    ## Loading the Dataset
    ds = load_sample()

    ## Printing Relevant Metadata tags

    attributes = [
        "Modality",
        "PixelSpacing",
        "WindowCenter",
        "WindowWidth",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "Rows",
        "Columns",
        "RescaleSlope",
        "RescaleIntercept",
        "BurnedInAnnotation",
    ]

    # DICOM data model is structured in a strict hierarchical level
    hierarchy_tags = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    ]

    # These tags carry demographic and clinical context information about the subject

    phi_tags = [
        "PatientName",
        "PatientBirthDate",
        "InstitutionName",
        "ReferringPhysicianName",
        "AccessionNumber",
        "StudyDate",
        "StudyTime",
        "Manufacturer",
        "DeviceSerialNumber",
    ]

    ## Geometry of DICOM Slice

    geometrical_tags = [
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "SliceThickness",
    ]

    print("---------- DICOM Metadata ----------")

    print_tags(ds, attributes)

    print("---------- Hierarchy Tags ----------")

    print_tags(ds, hierarchy_tags)

    print("---------- PHI Tags ---------------")

    print_tags(ds, phi_tags)

    print("---------- Geometry Tags ---------------")

    print_tags(ds, geometrical_tags)

    ## Number of Unique Pixel Values in Raw Image

    unique_val_raw = distinct_pixel_values(ds.pixel_array)

    print(f"\nDistinct Intensity Values before Windowing : {unique_val_raw}")

    ## Re-Scaling the Raw Pixels to Hounsfield Units

    print("\nProcessing Pixel Data ........")

    scaled_pixels = to_hounsfield_or_stored(ds)

    ## Getting the Window Center and Window Width Values from the Dataset If Available in the Metadata

    window_center, window_width = get_window(ds, scaled_pixels=scaled_pixels)

    ## Applying Window Scaling to the Scaled Pixel Values

    windowed_image = apply_window(
        scaled_pixels=scaled_pixels, center=window_center, width=window_width
    )

    unique_val_windowed = distinct_pixel_values(windowed_image)

    print(f"Distinct Intensity Values after Windowing : {unique_val_windowed}")

    ## Number of Pixels Clipped During Window scaling
    ## The Pixel Intensities above Value 1400 were clipped

    clipped_pixels = sum(
        count_outside_window(scaled_pixels, window_center, window_width)
    )

    print(f"The number of Pixels Clipped by Windowing: {clipped_pixels}")

    ## The Number of Unique Values if converted to 8 bit with no clipping

    full_range_8bit = scaled_quantization(scaled_pixels)

    print(
        f"Unique Pixel Values if 8-bit Quantization applied with no clipping: {len(np.unique_values(full_range_8bit))}"
    )

    ## Saving the Processed Image and the Raw Input Image

    save_fig(
        image=ds.pixel_array,
        title=f"Raw {ds.Modality}",
        out_path=out_path / "dicom_raw.png",
    )

    save_fig(
        windowed_image, title=ds.Modality, out_path=out_path / "dicom_windowed.png"
    )
    print("Image Successfully saved!!!!!!")


if __name__ == "__main__":
    main()
