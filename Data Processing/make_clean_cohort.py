import pandas as pd

EXCLUDE = [
    # failed hippocampal segmentation
    'OAS30041', 'OAS30234', 'OAS30241', 'OAS30379', 'OAS30662',
    'OAS30867', 'OAS30919', 'OAS31103', 'OAS31132',
    # corrupt source PET (single volume) -- Vo et al. exclude this too
    'OAS30065',
]

SRC = 'D:/mamba_model/thesis_cohort_final.csv'
DST = 'D:/mamba_model/thesis_cohort_clean.csv'

df = pd.read_csv(SRC)
clean = df[~df['subject_id'].isin(EXCLUDE)].reset_index(drop=True)
clean.to_csv(DST, index=False)
pd.DataFrame({'subject_id': EXCLUDE}).to_csv(
    'D:/mamba_model/excluded_subjects.csv', index=False)

print(f'{len(df)} -> {len(clean)} subjects')
print('class balance:', clean['outcome_label'].value_counts().to_dict())
