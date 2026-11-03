import matplotlib.pyplot as plt
import numpy as np
from pydicom.data import get_testdata_file

ds = get_testdata_file("MR_small.dcm", read=True)

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
]

for attribute in attributes:
    if hasattr(ds, attribute):
        val = ds.__getattr__(attribute)
        print(f"{attribute:<27}:{val} {type(val).__name__}")
    else:
        print(f"{attribute:<27}:Attribute not present")


# print(ds.PatientName)
# print(ds.Modality)
# print(f"{ds.Rows}X{ds.Columns}")
# print(ds.PhotometricInterpretation)
print(ds.pixel_array)  ## Numpy array

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.title("Sample DICOM MR Scan")
plt.show()

### Windowing the  Pixel Array so it can be read as 8 bit Image
window_center = float(ds.WindowCenter)
window_width = float(ds.WindowWidth)

## Clipping the Pixel Values
upper_bound = window_center + (window_width / 2)
lower_bound = window_center - (window_width / 2)

window_pixels = np.clip((ds.pixel_array).astype(np.float32), lower_bound, upper_bound)

# normalizing the Values
window_pixels = ((window_pixels - lower_bound) / (upper_bound - lower_bound)) * 255
window_pixels = window_pixels.astype(np.uint8)

plt.imshow(window_pixels, cmap=plt.cm.bone)
plt.title("Windowed CT Scan")
plt.show()
