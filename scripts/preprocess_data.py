# prep_3d_vols.py (robust version)
import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped, Compose
)
from monai.data import Dataset, DataLoader

def build_items(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["mask"].astype(str).str.len() > 0]
    items = []
    for _, r in df.iterrows():
        entry = {"subject": r["subject"], "mask": r["mask"]}
        for k in ("flair", "t2", "t1"):
            v = r.get(k, "")
            if isinstance(v, str) and v.strip():
                entry[k] = v
        if any(k in entry for k in ("flair", "t2", "t1")):
            items.append(entry)
    return items

def get_affine(batch, key_fallbacks):
    """Try to get affine from any available image key or mask."""
    def _as_numpy(affine):
        if hasattr(affine, "detach"):
            affine = affine.detach()
        if hasattr(affine, "cpu"):
            affine = affine.cpu()
        affine = np.asarray(affine)
        if affine.ndim == 3 and affine.shape[0] == 1:
            affine = affine[0]
        return affine

    for k in key_fallbacks:
        if k in batch:
            affine = getattr(batch[k], "affine", None)
            if affine is not None:
                return _as_numpy(affine)
        meta_key = f"{k}_meta_dict"
        if meta_key in batch and "affine" in batch[meta_key]:
            affine = batch[meta_key]["affine"]
            if isinstance(affine, (list, tuple)):
                affine = affine[0]
            return _as_numpy(affine)
    raise KeyError(f"No affine found for keys {key_fallbacks}")

def save_nifti_from_batch(batch, key, out_path: Path, affine):
    """Save a single key from MONAI batch using nibabel."""
    if key not in batch:
        return
    data = batch[key][0].detach().cpu().numpy()
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if key == "mask":
        data = (data > 0.5).astype(np.uint8)
    else:
        data = data.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(out_path))

def main(root, manifest, outdir):
    root = Path(root)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    items = build_items(manifest)

    # determine available keys
    present = set()
    for it in items:
        present |= set(it.keys())
    img_keys = [k for k in ("flair", "t2", "t1") if k in present]
    keys = img_keys + ["mask"]
    modes = tuple("nearest" if k == "mask" else "bilinear" for k in keys)

    tx = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=modes),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=img_keys, a_min=0, a_max=3000,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key=img_keys[0] if img_keys else "mask", margin=8),
        EnsureTyped(keys=keys),
    ])

    ds = Dataset(items, tx)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    for batch in tqdm(dl, total=len(ds)):
        subj = batch["subject"][0]
        subdir = outdir / subj
        # use first available affine as reference
        affine = get_affine(batch, img_keys + ["mask"])
        for k in img_keys:
            save_nifti_from_batch(batch, k, subdir / f"{k}.nii.gz", affine)
        save_nifti_from_batch(batch, "mask", subdir / "mask.nii.gz", affine)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Path to NIfTI-files dir")
    ap.add_argument("--manifest", default="/home/j98my/Pre-cleaning/csv's/gbm_manifest3d_masked.csv")
    ap.add_argument("--outdir", default="prep/gbm")
    args = ap.parse_args()
    main(args.root, args.manifest, args.outdir)
