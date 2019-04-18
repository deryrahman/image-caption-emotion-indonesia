from nltk.translate.bleu_score import corpus_bleu


def bleu_evaluator(captions_listlist, predictions_list):
    captions_listlist = [[caption.split(' ')
                          for caption in captions_list]
                         for captions_list in captions_listlist]
    predictions_list = [
        prediction.split(' ') for prediction in predictions_list
    ]
    bleu_1 = corpus_bleu(
        list_of_references=captions_listlist,
        hypotheses=predictions_list,
        weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(
        list_of_references=captions_listlist,
        hypotheses=predictions_list,
        weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(
        list_of_references=captions_listlist,
        hypotheses=predictions_list,
        weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(
        list_of_references=captions_listlist,
        hypotheses=predictions_list,
        weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_1, bleu_2, bleu_3, bleu_4
