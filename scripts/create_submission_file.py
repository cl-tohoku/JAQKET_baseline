import os
import sys
import json
import argparse
from typing import List, Dict

# dict_keys(['qid', 'question', 'answer_entity', 'answer_candidates', 'original_question'])
Candidate = Dict[str, str]  


def list_fm_jsonl(f_jsonl: os.path.abspath) -> List[Candidate]:
    """ jsonl -> List[Dict[str, str]] """
    return [json.loads(line.rstrip()) for line in open(f_jsonl, 'r')]


def list_fm_tsv(f_tsv: os.path.abspath, col=0) -> List[int]:
    """ 2cols (pred, out_label_id) -> List[pred:int] """
    return [int(line.split()[col]) for line in open(f_tsv, 'r')]


def create_arg_parser():
    """
    ### required
    - test_question_file    (os.path.abspath)   path of a test set
    - pred_label_file       (os,path.abspath)   path of a predicted file
    
    ### optional
    - output_file           (os.path.abspath)   path of output file
    - w_question            (bool)              write with question
    - w_candidates          (bool)              write with candidates
    """
    parser = argparse.ArgumentParser(description=
            'Read a predicted file and Convert into a submission format')

    parser.add_argument(
        '-test', '--test_question_file',
        dest='test',
        required=True,
        type=os.path.abspath,
        help='test file: "aio_leaderboard.json"'
    )
    parser.add_argument(
        '-pred', '--pred_label_file',
        dest='pred',
        required=True,
        type=os.path.abspath,
        help='predicted file: "is_test_true_output_labels.txt"'
    )
    parser.add_argument(
        '-fo', '--output_file',
        dest='fo',
        type=os.path.abspath,
        default=os.path.abspath('submission.json'),
        help='submission file: "submission.json"'
    )
    parser.add_argument(
        '-wq', '--with_question',
        action='store_const',
        const='question',
        help='write w/ question if necessary'
    )
    parser.add_argument(
        '-wc', '--with_candidates',
        action='store_const',
        const='answer_candidates',
        help='write w/ candidates if necessary'
    )

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    answers = list_fm_jsonl(args.test)    # List[Candidate]
    pred_labels = list_fm_tsv(args.pred)  # List[int]

    #assert len(answers) == len(pred_labels)
    
    fo = open(args.fo, 'w')

    for answer_info, pred_label in zip(answers, pred_labels):
        result = {
                  'qid': answer_info['qid'],
                  'answer_entity': answer_info['answer_candidates'][pred_label]
                 }
        
        if args.with_question is not None:
            result['question'] = answer_info['question']

        if args.with_candidates is not None:
            result['answer_candidates'] = answer_info['answer_candidates']

        json.dump(result, fo, ensure_ascii=False)
        fo.write('\n')

    fo.close()
    sys.stdout.write('create submission file: %s\n' % args.fo)


if __name__ == '__main__':
    """ running example
    $ python make_result_file.py \
            -test qio_leaderboard.json \
            -pred is_test_true_output_labels.txt \
            -fo submission.json \
            -wq \
            -wc
    """
    main()
