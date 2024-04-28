from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


predicted_answers = []
real_answers = []

rouge = Rouge()
rouge_scores = rouge.get_scores(predicted_answers, real_answers, avg=True)
print("ROUGE scores:", rouge_scores)

# BLEU
smoothing_function = SmoothingFunction().method1  # Define smoothing function
bleu_score = corpus_bleu([[ref.split()] for ref in predicted_answers], real_answers.split(), smoothing_function=smoothing_function)
print("BLEU score:", bleu_score)

# Calculate EM
em_score = 0
for i in range(0,len(predicted_answers)): 
    if predicted_answers[i] == real_answers[i]:
        em_score += 1

print("Exact Match (EM) score:", em_score/len(predicted_answers))
