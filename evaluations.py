from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


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
if predicted_answers == ground_truth_answera:
    em_score = 1.0
else:
    em_score = 0.0

print("Exact Match (EM) score:", em_score)
