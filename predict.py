import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel


def predict_for_file(test_data, model, batch_size=32, to_normalize=False):
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent)
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    return cnt_corrections,result_lines


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    cnt_corrections = predict_for_file(args.input_data, args.output_file, model,
                                       batch_size=args.batch_size, 
                                       to_normalize=args.normalize)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")
