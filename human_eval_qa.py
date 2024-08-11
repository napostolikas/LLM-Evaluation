import pandas as pd
from sklearn.metrics import cohen_kappa_score


df_1 = pd.read_excel('./human_eval_QA_Camilo.xlsx')
df_2 = pd.read_excel('./human_eval_QA_Yi.xlsx')
df_3 = pd.read_excel('./human_eval_QA_Kinga.xlsx')
df_4 = pd.read_excel('./human_eval_QA_Nikos.xlsx')

camilo_answer = sum(list(df_1['Your Answer']))
yi_answer = sum(list(df_2['Your Answer']))
kinga_answer = sum(list(df_3['Your Answer']))
nikos_answer = sum(list(df_4['Your Answer']))

socratic_answer = sum(list(df_3["Conversation's result"]))

camilo_acc = camilo_answer / len(df_1)
yi_acc = yi_answer / len(df_2)
kinga_acc = kinga_answer / len(df_3)
nikos_acc = nikos_answer / len(df_4)

socratic_acc = socratic_answer / len(df_3)

print(f"Camilo: {camilo_acc}")
print(f"Yi: {yi_acc}")
print(f"Kinga: {kinga_acc}")
print(f"Nikos: {nikos_acc}")

print(f"Socratic: {socratic_acc}")


kappa1 = cohen_kappa_score(list(df_1['Your Answer']), list(df_2['Your Answer']))
kappa2 = cohen_kappa_score(list(df_1['Your Answer']), list(df_3['Your Answer']))
kappa3 = cohen_kappa_score(list(df_1['Your Answer']), list(df_4['Your Answer']))
kappa4 = cohen_kappa_score(list(df_2['Your Answer']), list(df_3['Your Answer']))
kappa5 = cohen_kappa_score(list(df_2['Your Answer']), list(df_4['Your Answer']))
kappa6 = cohen_kappa_score(list(df_3['Your Answer']), list(df_4['Your Answer']))

average_kappa_score = (kappa1 + kappa2 + kappa3 + kappa4 + kappa5 + kappa6) / 6

print(f"Average Kappa Score between annotators: {average_kappa_score}")

kappa7 = cohen_kappa_score(list(df_1['Your Answer']), list(df_4["Conversation's result"]))
kappa8 = cohen_kappa_score(list(df_2['Your Answer']), list(df_4["Conversation's result"]))
kappa9 = cohen_kappa_score(list(df_3['Your Answer']), list(df_4["Conversation's result"]))
kappa10 = cohen_kappa_score(list(df_4['Your Answer']), list(df_4["Conversation's result"]))

print(f"Camilo vs Socratic: {kappa7}")
print(f"Yi vs Socratic: {kappa8}")
print(f"Kinga vs Socratic: {kappa9}")
print(f"Nikos vs Socratic: {kappa10}")

average_kappa_score_socratic = (kappa7 + kappa8 + kappa9 + kappa10) / 4
print(f"Average Kappa Score between annotators and Socratic: {average_kappa_score_socratic}")

