import pandas as pd
from sklearn.metrics import cohen_kappa_score


df_1 = pd.read_excel('./human_eval_NER_Camilo.xlsx')
df_2 = pd.read_excel('./human_eval_NER_Yi.xlsx')
df_3 = pd.read_excel('./human_eval_NER_Kinga.xlsx')
df_4 = pd.read_excel('./human_eval_NER_Nikos.xlsx')

fixed_phrases = ['are no', 'is no', 'does not', 'N/A']

for i in range(len(df_1)):
    for phrase in fixed_phrases:
        if (phrase in df_1['Candidate Answer'][i]) or (df_1['Your FN'][i] > 40):
            df_1.drop(i, inplace=True)
            break

for i in range(len(df_2)):
    for phrase in fixed_phrases:
        if phrase in df_2['Candidate Answer'][i] or (df_2['Your FN'][i] > 40):
            df_2.drop(i, inplace=True)
            break

for i in range(len(df_3)):
    for phrase in fixed_phrases:
        if phrase in df_3['Candidate Answer'][i] or (df_3['Your FN'][i] > 40):
            df_3.drop(i, inplace=True)
            break

for i in range(len(df_4)):
    for phrase in fixed_phrases:
        if phrase in df_4['Candidate Answer'][i] or (df_4['Your FN'][i] > 40):
            df_4.drop(i, inplace=True)
            break

camilo_tp = list(df_1['Your TP'])
camilo_fp = list(df_1['Your FP'])
camilo_fn = list(df_1['Your FN'])

yi_tp = list(df_2['Your TP'])
yi_fp = list(df_2['Your FP'])
yi_fn = list(df_2['Your FN'])

kinga_tp = list(df_3['Your TP'])
kinga_fp = list(df_3['Your FP'])
kinga_fn = list(df_3['Your FN'])

nikos_tp = list(df_4['Your TP'])
nikos_fp = list(df_4['Your FP'])
nikos_fn = list(df_4['Your FN'])

# for i in range(len(camilo_tp)):
#     print(f'Iteration {i}')
#     print(f'Camilo: {camilo_tp[i]} | Yi: {yi_tp[i]} | Kinga: {kinga_tp[i]} | Nikos: {nikos_tp[i]}')
#     print(f'Camilo: {camilo_fp[i]} | Yi: {yi_fp[i]} | Kinga: {kinga_fp[i]} | Nikos: {nikos_fp[i]}')
#     print(f'Camilo: {camilo_fn[i]} | Yi: {yi_fn[i]} | Kinga: {kinga_fn[i]} | Nikos: {nikos_fn[i]}')
#     print("--------------------")

socratic_tp = df_4['TP'].sum()
socratic_fp = df_4['FP'].sum()
socratic_fn = df_4['FN'].sum()

tp1 = df_1['Your TP'].sum()
fp1 = df_1['Your FP'].sum()
fn1 = df_1['Your FN'].sum()

tp2 = df_2['Your TP'].sum()
fp2 = df_2['Your FP'].sum()
fn2 = df_2['Your FN'].sum()

tp3 = df_3['Your TP'].sum()
fp3 = df_3['Your FP'].sum()
fn3 = df_3['Your FN'].sum()

tp4 = df_4['Your TP'].sum()
fp4 = df_4['Your FP'].sum()
fn4 = df_4['Your FN'].sum()

precision1 = tp1 / (tp1 + fp1)
recall1 = tp1 / (tp1 + fn1)
f1_score1 = 2 * (precision1 * recall1) / (precision1 + recall1)

precision2 = tp2 / (tp2 + fp2)
recall2 = tp2 / (tp2 + fn2)
f1_score2 = 2 * (precision2 * recall2) / (precision2 + recall2)

precision3 = tp3 / (tp3 + fp3)
recall3 = tp3 / (tp3 + fn3)
f1_score3 = 2 * (precision3 * recall3) / (precision3 + recall3)

precision4 = tp4 / (tp4 + fp4)
recall4 = tp4 / (tp4 + fn4)
f1_score4 = 2 * (precision4 * recall4) / (precision4 + recall4)

socratic_precision = socratic_tp / (socratic_tp + socratic_fp)
socratic_recall = socratic_tp / (socratic_tp + socratic_fn)
socratic_f1_score = 2 * (socratic_precision * socratic_recall) / (socratic_precision + socratic_recall)

print("\n")
print(f'Precision for Camilo: {precision1}')
print(f'Recall for Camilo: {recall1}')
print(f'F1-score for Camilo: {f1_score1}')
print("\n")
print(f'Precision for Yi: {precision2}')
print(f'Recall for Yi: {recall2}')
print(f'F1-score for Yi: {f1_score2}')
print("\n")
print(f'Precision for Kinga: {precision3}')
print(f'Recall for Kinga: {recall3}')
print(f'F1-score for Kinga: {f1_score3}')
print("\n")
print(f'Precision for Nikos: {precision4}')
print(f'Recall for Nikos: {recall4}')
print(f'F1-score for Nikos: {f1_score4}')
print("\n")
print(f'Precision for Socratic: {socratic_precision}')
print(f'Recall for Socratic: {socratic_recall}')
print(f'F1-score for Socratic: {socratic_f1_score}')
print("\n")


kappa1 = cohen_kappa_score(df_1['Your TP'], df_2['Your TP'])
kappa2 = cohen_kappa_score(df_1['Your TP'], df_3['Your TP'])
kappa3 = cohen_kappa_score(df_1['Your TP'], df_4['Your TP'])
kappa4 = cohen_kappa_score(df_2['Your TP'], df_3['Your TP'])
kappa5 = cohen_kappa_score(df_2['Your TP'], df_4['Your TP'])
kappa6 = cohen_kappa_score(df_3['Your TP'], df_4['Your TP'])

average_kappa = (kappa1 + kappa2 + kappa3 + kappa4 + kappa5 + kappa6) / 6

print(f'TP: Average kappa score between annotators: {average_kappa}')
print("\n")

kappa1 = cohen_kappa_score(df_1['Your FP'], df_2['Your FP'])
kappa2 = cohen_kappa_score(df_1['Your FP'], df_3['Your FP'])
kappa3 = cohen_kappa_score(df_1['Your FP'], df_4['Your FP'])
kappa4 = cohen_kappa_score(df_2['Your FP'], df_3['Your FP'])
kappa5 = cohen_kappa_score(df_2['Your FP'], df_4['Your FP'])
kappa6 = cohen_kappa_score(df_3['Your FP'], df_4['Your FP'])

average_kappa = (kappa1 + kappa2 + kappa3 + kappa4 + kappa5 + kappa6) / 6
print(f'FP: Average kappa score between annotators: {average_kappa}')
print("\n")

kappa1 = cohen_kappa_score(df_1['Your FN'], df_2['Your FN'])
kappa2 = cohen_kappa_score(df_1['Your FN'], df_3['Your FN'])
kappa3 = cohen_kappa_score(df_1['Your FN'], df_4['Your FN'])
kappa4 = cohen_kappa_score(df_2['Your FN'], df_3['Your FN'])
kappa5 = cohen_kappa_score(df_2['Your FN'], df_4['Your FN'])
kappa6 = cohen_kappa_score(df_3['Your FN'], df_4['Your FN'])

average_kappa = (kappa1 + kappa2 + kappa3 + kappa4 + kappa5 + kappa6) / 6
print(f'FN: Average kappa score between annotators: {average_kappa}')
print("\n")

kappa_camilo_tp = cohen_kappa_score(df_1['Your TP'], df_4['TP'])
kappa_yi_tp = cohen_kappa_score(df_2['Your TP'], df_4['TP'])
kappa_kinga_tp = cohen_kappa_score(df_3['Your TP'], df_4['TP'])
kappa_nikos_tp = cohen_kappa_score(df_4['Your TP'], df_4['TP'])

average_kappa_tp = (kappa_camilo_tp + kappa_yi_tp + kappa_kinga_tp + kappa_nikos_tp) / 4

print(f"Average kappa score between annotators and Socratic: {average_kappa_tp}")

kappa_camilo_fp = cohen_kappa_score(df_1['Your FP'], df_4['FP'])
kappa_yi_fp = cohen_kappa_score(df_2['Your FP'], df_4['FP'])
kappa_kinga_fp = cohen_kappa_score(df_3['Your FP'], df_4['FP'])
kappa_nikos_fp = cohen_kappa_score(df_4['Your FP'], df_4['FP'])

average_kappa_fp = (kappa_camilo_fp + kappa_yi_fp + kappa_kinga_fp + kappa_nikos_fp) / 4

print(f"Average kappa score between annotators and Socratic: {average_kappa_fp}")

kappa_camilo_fn = cohen_kappa_score(df_1['Your FN'], df_4['FN'])
kappa_yi_fn = cohen_kappa_score(df_2['Your FN'], df_4['FN'])
kappa_kinga_fn = cohen_kappa_score(df_3['Your FN'], df_4['FN'])

average_kappa_fn = (kappa_camilo_fn + kappa_yi_fn + kappa_kinga_fn) / 3

print(f"Average kappa score between annotators and Socratic: {average_kappa_fn}")




