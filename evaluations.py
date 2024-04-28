import json
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

with open('results/qa_checkpoint-40.json', 'r') as f:
    answers = json.load(f)

predicted_answers = [s['pred'].lower() for s in answers]
real_answers = [s['real'].lower() for s in answers]

rouge = Rouge()
rouge_scores = rouge.get_scores(predicted_answers, real_answers, avg=True)
print("ROUGE scores:", rouge_scores)

# BLEU
smoothing_function = SmoothingFunction().method1  # Define smoothing function
bleu_score = corpus_bleu(predicted_answers, real_answers, smoothing_function=smoothing_function)
print("BLEU score:", bleu_score)

# Calculate EM
em_score = 0
for i in range(0,len(predicted_answers)): 
    if predicted_answers[i] == real_answers[i]:
        em_score += 1

print("Exact Match (EM) score:", em_score/len(predicted_answers))
