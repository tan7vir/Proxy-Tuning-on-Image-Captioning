#@title get_the_scores Method
import sys
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

def get_the_scores(reference, candidate):
    reference = {'image_id': ['2sfdbs4646'], 'text': reference}
    candidate = {'image_id': ['2sfdbs4646'], 'text': candidate}

    # Calculate CIDEr score
    
    cider_calculator = Cider()
    cider_score, cider_scores = cider_calculator.compute_score(reference, candidate)

    bleu_calculator = Bleu()
    bleu_score, bleu_scores = bleu_calculator.compute_score(reference, candidate)

    rouge_calculator = Rouge()
    rouge_score, rouge_scores = rouge_calculator.compute_score(reference, candidate)

    # Return scores
    return {
        'CIDEr': {'score': cider_score, 'scores': cider_scores},
        'BLEU': {'score': bleu_score, 'scores': bleu_scores},
        'Rouge': {'score': rouge_score, 'scores': rouge_scores}
    }
