import pandas as pd
import os

df = pd.read_csv('D:/mamba_model/thesis_cohort_final.csv')
pet_dir = 'E:/pet scans/'
existing_pet = os.listdir(pet_dir)

exact_match = []
close_match = []
no_match = []

for _, row in df.iterrows():
    subj = row['subject_id']
    pet_day = int(row['pet_day'])
    
    matches = [f for f in existing_pet 
               if f.startswith(subj) and 'PIB' in f and not f.endswith('.zip')]
    
    if not matches:
        no_match.append({'subject_id': subj, 'pet_day': pet_day, 'available': 'NONE'})
        continue
    
    # Find closest
    best = None
    best_diff = 9999
    for m in matches:
        try:
            d = int(m.split('_d')[1])
            diff = abs(d - pet_day)
            if diff < best_diff:
                best_diff = diff
                best = m
        except:
            continue
    
    if best_diff == 0:
        exact_match.append(subj)
    elif best_diff <= 30:
        close_match.append({'subject_id': subj, 'pet_day': pet_day, 
                            'best_folder': best, 'day_diff': best_diff})
    else:
        no_match.append({'subject_id': subj, 'pet_day': pet_day, 
                        'best_available': best, 'day_diff': best_diff})

print(f'Exact match:  {len(exact_match)}')
print(f'Close match (<=30 days): {len(close_match)}')
print(f'Poor/no match (>30 days): {len(no_match)}')

print('\nClose matches:')
for r in close_match:
    print(f"  {r['subject_id']}: need d{r['pet_day']}, have {r['best_folder']} (diff={r['day_diff']})")

print('\nPoor/no matches:')
for r in no_match:
    print(f"  {r['subject_id']}: need d{r['pet_day']}, best={r.get('best_available','NONE')} (diff={r.get('day_diff','N/A')})")