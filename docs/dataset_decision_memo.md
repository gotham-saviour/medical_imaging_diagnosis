# Dataset Decision Memo: Medical Diagnosis Project

**Topic:** Evaluation and Selection of Classification and Segmentation Datasets

---

## 1. Classification Datasets Evaluation

To select the primary dataset for thoracic disease classification, we evaluated two industry-standard benchmarks: **NIH ChestX-ray8 / ChestX-ray14** and **CheXpert**.

### Summary Comparison

| Feature / Metric | NIH ChestX-ray14 | CheXpert (Stanford) |
| :--- | :--- | :--- |
| **Unique Patients** | 30,805 (ChestX-ray8: 32,717 unique patients) | 65,240 |
| **Total Images** | 112,120 frontal-view images (ChestX-ray8: 108,948) | 224,316 frontal and lateral views |
| **Format & Resolution** | PNG (1024 x 1024, Greyscale) | JPEG/PNG (1024 x 1024 downsampled) |
| **Observations** | 14 common thoracic diseases + "No Finding" (expanded from 8 in ChestX-ray8) | 14 observations (pathological & non-disease) |
| **Label Encoding** | Binary (0 and 1) | Ternary (0, 1, and -1 for uncertain) |
| **Dataset Size** | ~42 GB | ~440 GB (Full resolution) |
| **License & Terms** | Unrestricted / Public Domain (NIH Clinical Center) | Non-Commercial Research Only / No Redistribution |
| **Access Method** | Direct download via NIH/Kaggle | Credentialed gateway / Downsampled on Kaggle |

---

### 1.1 NIH ChestX-ray8 / ChestX-ray14
*Note: Originally released as "ChestX-ray8" with 8 thoracic pathology classes, later expanded to "ChestX-ray14" covering 14 common classes.*

* **Patient Cohort:** 30,805 unique patients (32,717 in the initial ChestX-ray8 release) collected from the NIH Clinical Center.
* **Imaging Specs:** 112,120 sample greyscale chest X-ray images (frontal view only: AP and PA) stored in 8-bit PNG format at 1024 x 1024 resolution.
* **Observations:** Extracted from institutional radiology reports using NLP tools (DNorm and MetaMap with syntactic dependency negation rules) to generate multi-label tags while filtering out negations and uncertainties.
* **Labels:** Binary (0 and 1) for each pathology category.
* **Dataset Size:** ~42 GB.
* **License & Terms:** Unrestricted public domain hosting by the NIH Clinical Center.
* **Known Issues & Planned Mitigations:**
  * **Class Imbalance:** Over 50% of images are labeled "Normal/No-Findings," while rare classes like Hernia account for roughly 0.20% of the dataset.
    * *Mitigation:* Implement class-weighted Binary Cross-Entropy (BCE) loss and positive/negative instance balancing factors during training.
  * **Splitting Rule Constraint:** Longitudinal scans result in multiple images per patient (averaging ~3.6 scans per patient), risking data leakage if split at the image level.
    * *Mitigation:* Enforce strict patient-level splitting (70% train, 10% validation, 20% test) grouped by unique patient identifiers.
  * **Label Noise:** Automated NLP extraction introduces systematic label noise due to clinical misspellings and complex syntactic constructs.
    * *Mitigation:* Apply label smoothing and syntactic-level negation/uncertainty rules during data preprocessing.

---

### 1.2 CheXpert Dataset
*Developed by the Stanford Machine Learning Group using chest radiographs from Stanford Hospital.*

* **Patient Cohort:** 65,240 unique patients.
* **Imaging Specs:** 224,316 chest radiographs in JPEG format containing both frontal and lateral views.
* **Observations:** 14 observations including pathological findings and non-disease clinical features (such as Support Devices and Fractures), extracted via a three-stage rule-based NLP pipeline.
* **Labels:** Ternary encoding containing positive (1), negative (0), and uncertain (-1) labels.
* **Dataset Size:** ~440 GB (full resolution).
* **License & Terms:** Restricted to non-commercial research purposes with no redistribution permitted; requires a credentialed download gateway.
* **Known Issues:**
  * **Class Imbalance:** High prevalence of common findings like Support Devices (56.40%) and Lung Opacity (49.39%), contrasted with rare pathologies.
  * **Uncertainty Prevalence:** Uncertain (-1) labels are frequent (e.g., 12.78% for Consolidation), requiring specialized handling strategies (such as U-Ignore, U-Ones, U-Zeros, or U-MultiClass).

---

## 2. Segmentation Task: Medical Segmentation Decathlon (MSD)

* **Patient Cohort:** 2,633 expert-annotated 3D volumetric scans across 10 distinct anatomical tasks.
* **Labels:** 3D volumetric masks with voxel values marked as 1 for the target organ and 0 for the background.
* **Dimensions:** 3D volumetric scans with typical axial dimensions around 512 x 512 x Z (e.g., ~120 slices).
* **Dataset Size:** ~76 GB total; scoped down locally to **Task 09 Spleen (~1.6 GB)** and **Task 02 Heart (~0.4 GB)**.
* **Data Format:** Formatted as compressed Neuroimaging Informatics Technology Initiative (NIfTI) files (`.nii.gz`).
* **Scoped Tasks & Modalities:**
  * **Task 09 (Spleen - Primary):** Mono-modal contrast-enhanced CT (`CT` modality) targeting spleen segmentation (61 volumes: 41 training, 20 testing).
  * **Task 02 (Heart - Fallback):** Mono-modal MRI (`MR` modality) targeting left atrium segmentation (30 volumes: 20 training, 10 testing).
* **Licensing:** Distributed openly under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
* **Known Issues & Planned Mitigations:**
  * **Unbalanced Voxel Distribution:** Foreground organ voxels occupy a minor fraction of the 3D volume compared to the background.
    * *Mitigation:* Utilize Dice Loss (or combined Dice + Cross-Entropy Loss) and patch-based foreground sampling.
  * **High Image Variations:** Multi-site variations in contrast, intensity scales, and scanner protocols.
    * *Mitigation:* Standardize preprocessing using Z-score intensity normalization for MRI and Hounsfield Unit (HU) windowing/clipping for CT.

---

## 3. Infrastructure & Engineering Constraints

### 3.1 DICOM and Orthanc Compatibility
* **Requirement:** Convert curations of native 2D PNG radiographs and 3D NIfTI volumes into structured DICOM files (`.dcm`) to interface smoothly with an Orthanc PACS server.
* **Metadata Injection:** Mandates compliance headers containing required identifiers and spatial tags:
  * Core Study/Series Tags: `PatientID`, `StudyInstanceUID`, `SeriesInstanceUID`, `Modality` (`DX`/`CR`, `MR`, `CT`), `SOPClassUID`, `SOPInstanceUID`.
  * Spatial Geometry Tags (for 3D volumes): `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, `SliceThickness`, `FrameOfReferenceUID`.
* **Implementation:** Automated ingestion pipeline via `pydicom` and `nibabel` Python libraries.

### 3.2 Compute Constraints
Operating under local resource limits (Drive D: 86.7 GB free, Drive C: 6 GB free, 8 GB RAM with 3.8 GiB visible to WSL, no local GPU) alongside a remote Kaggle Tesla T4 GPU:
1. **Downsampled Data Subsets:**
   * **Classification Subset:** Utilize a patient-stratified subset of 15,000 images from NIH ChestX-ray14 via Rare-Positive Anchoring to preserve rare class support while maintaining patient-level isolation.
   * **Segmentation Subset:** Restrict downloads strictly to **Task 09 Spleen** and **Task 02 Heart** to conserve local storage.
2. **Hybrid Cloud Training:** Develop and debug pipelines locally using CPU-only PyTorch builds, executing heavy training jobs on Kaggle Tesla T4 GPUs.

---

## 4. Final Dataset Selection (Classification)

We select the **NIH ChestX-ray14** dataset over CheXpert because:
1. **Streamlined Workflow:** Direct public availability via Kaggle/NIH avoids credentialed approval barriers and fits local storage constraints (~42 GB vs. ~440 GB).
2. **Legal and Commercial Freedom:** Unrestricted public domain licensing allows unfettered downstream distribution.
3. **Loss Function Simplicity:** Strictly binary labels (0 and 1) allow direct application of standard Binary Cross-Entropy (BCE) loss without custom uncertainty-mapping heuristics.

### Trade-offs Accepted by Rejecting CheXpert:
1. Loss of lateral radiographic views.
2. Absence of radiologist-adjudicated consensus test/validation benchmarks.
3. Higher susceptibility to automated NLP parsing label noise.

---

## 5. Risks & Mitigations

| Risk Category | Risk Description | Impact | Planned Mitigation |
| :--- | :--- | :--- | :--- |
| **Prevalence Shift via Rare-Positive Anchoring** | Artificially boosting rare disease representation in the local subset shifts base probability distributions away from the full dataset, potentially miscalibrating model outputs. | Medium | Rely on threshold-independent ranking metrics (AUROC/AUPRC) locally; perform temperature scaling and bias recalibration on the natural distribution during full cloud training. |
| **3D Geometry & Voxel Spacing Loss in DICOM Conversion** | Translating NIfTI affine matrices into custom DICOM headers can cause coordinate inversions or spatial distortion. | High | Programmatically parse 4x4 affine matrices using `nibabel` to write valid spatial geometry tags; execute round-trip validation tests. |
| **WSL Memory Overload (OOM)** | Low WSL RAM ceiling (3.8 GiB) risks out-of-memory crashes when loading full 3D volumes or local Orthanc Docker containers. | High | Set `num_workers=0` or `1` in PyTorch DataLoaders, employ memory-mapped NIfTI reading proxies, and host all container images on Drive D:. |

---

## 6. Open Questions

1. **Namespace Isolation in Orthanc:** Should NIH 2D CXR studies and MSD 3D volumes be isolated under distinct namespace prefixes in Orthanc or mapped to unified synthetic patient records?
2. **Model Topology Selection for Segmentation:** Given local memory restrictions, should the segmentation pipeline utilize 2D U-Net slice processing or memory-efficient 3D sub-volume patch extraction?
3. **Loss Weighting Co-Occurrence Strategy:** For heavily co-occurring pathologies in NIH ChestX-ray14, should positive class weights (`pos_weight`) use strict marginal inverse frequencies or a damped square-root scale?

---

## 7. References

1. **Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).** *ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* IEEE CVPR, pp. 2097–2106.
   * **p. 2097:** Introduction of ChestX-ray8, 32,717 unique patients, 108,948 frontal images, and public domain release.
   * **p. 2098–2099:** NLP text-mining pipeline details (DNorm, MetaMap, Bllip parser, syntactic negation/uncertainty rules) and image preprocessing.
   * **p. 2104 (Appendix):** Expansion to ChestX-ray14 covering 14 common diseases, including class frequencies (e.g., Hernia at 227 images, ~0.20%) and Normal findings (>50%).

2. **Irvin, J., et al. (2019).** *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI, Vol. 33(01), pp. 590–597.
   * **p. 590–591:** Dataset scale (224,316 radiographs, 65,240 patients), frontal/lateral views, 14 observations, and baseline distribution (Table 1).
   * **p. 591–593:** Rule-based 3-stage NLP labeler, ternary label encoding (0, 1, -1), and evaluation against radiologist consensus panels.

3. **Antonelli, M., et al. (2022).** *The Medical Segmentation Decathlon.* Nature Communications, 13(1), Article 4128, pp. 1–13.
   * **p. 1–2:** Multi-task benchmark overview, unbalance challenges, and evaluation methodology (Dice and NSD metrics).
   * **p. 9:** Specifications for Task 02 Heart (MRI left atrium) and Task 09 Spleen (CT spleen), NIfTI format rules, and CC BY-SA 4.0 licensing.
