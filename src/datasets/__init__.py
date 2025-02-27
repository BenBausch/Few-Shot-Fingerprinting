import glob
import os


def create_omniglot_subject(subject_path, file_endings):
    """
    Each character in Omniglot dataset has 20 images. Each character is a subject.

    The subject file names are in the format:
    "Sanskrit/character01/0709_01.png", "Sanskrit/character01/0709_02.png", "Sanskrit/character01/0709_03.png", ...
    """
    subject_files = glob.glob(os.path.join(subject_path, f"*{file_endings}"))

    subject_id = subject_path.split("/")[-2] + "/" + subject_path.split("/")[-1]

    sorted_s_files = sorted(
        subject_files, key=lambda x: int(os.path.basename(x)[-6:-4])
    )

    subject = {
        **{"id": subject_id},
        **{"drawing": [s_f for s_f in sorted_s_files]},
        "modalities": ["drawing"],
        "meta_data": None,
    }

    return subject


def create_xray_subject(subject_path, file_endings, meta_data=None):
    """
    Each subject in the X-ray dataset has multiple images.

    The subject file names are in the format:
    "path/to/folder/xray/images/subject_id/img-num.png"
    "path/to/folder/xray/images/00000001/000.png",
    "path/to/folder/xray/images/00000001/001.png",
    "path/to/folder/xray/images/00000001/002.png", ...

    Patients have at least 1 image.
    """
    patient_id = subject_path.split("/")[-1]

    patient_files = glob.glob(os.path.join(subject_path, f"*{file_endings}"))

    # Sort files by image number
    sorted_s_files = sorted(patient_files, key=lambda x: int(os.path.basename(x)[:-4]))

    subject = {"id": patient_id, "xray": sorted_s_files, "modalities": ["xray"]}

    # Add metadata per image if available
    if meta_data is not None:
        subject["meta_data"] = {"xray": {}}
        for idx, img in enumerate(sorted_s_files):
            key = patient_id + "_" + os.path.basename(img)
            subject["meta_data"]["xray"][idx] = meta_data[
                meta_data["Image Index"] == key
            ].to_dict("records")[0]

    return subject


def create_mri_subject(subject_path, file_endings):
    """
    Each subject in the MRI dataset has multiple images.

    The subject file names are in the format:
    "path/to/folder/ptn-num/BraTS2021_ptn-num_modality.nii.gz" Ex; "path/to/folder/00052/BraTS2021_00052_flair.nii.gz", ....

    Patients have 4 images, 1 of each mri modality (t1, t1ce, t2, flair).
    """

    patient_id = subject_path.split("/")[-1]

    patient_files = glob.glob(os.path.join(subject_path, f"*{file_endings}"))

    subject = {
        "id": patient_id,
        "t1": [f for f in patient_files if "t1" in f and "t1ce" not in f],
        "t1ce": [f for f in patient_files if "t1ce" in f],
        "t2": [f for f in patient_files if "t2" in f],
        "flair": [f for f in patient_files if "flair" in f],
        "modalities": ["t1", "t1ce", "t2", "flair"],
        "meta_data": None,
    }

    return subject
