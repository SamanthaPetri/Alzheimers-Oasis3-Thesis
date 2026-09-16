import nibabel as nib
import os
import pandas as pd

df = pd.read_csv('D:/mamba_model/thesis_cohort_final.csv')
fs_output = 'E:/Oasis3/FastSurfer_output'

done = 0
skipped = 0
failed = 0

for _, row in df.iterrows():
    session = row['mri_session']
    mgz_path = f'{fs_output}/{session}/mri/orig.mgz'
    nii_path = f'{fs_output}/{session}/mri/orig.nii.gz'

    if os.path.exists(nii_path):
        skipped += 1
        continue

    if not os.path.exists(mgz_path):
        print(f'MISSING: {session}')
        failed += 1
        continue

    try:
        img = nib.load(mgz_path)
        nib.save(nib.Nifti1Image(img.get_fdata(), img.affine), nii_path)
        done += 1
        print(f'Converted {session} ({done})')
    except Exception as e:
        print(f'FAILED {session}: {e}')
        failed += 1

print(f'\nDone: {done}, Skipped: {skipped}, Failed: {failed}')