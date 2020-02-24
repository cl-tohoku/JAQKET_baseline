# coding=utf-8

from __future__ import absolute_import, division, print_function

import json
import tqdm
import argparse
import glob
import logging
import os
import random
import numpy as np
import torch

from typing import List
from io import open
import gzip
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForMultipleChoice,
    BertJapaneseTokenizer,
    PreTrainedTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

###############################################################################
###############################################################################


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence
                      (context of corresponding question).
            question: string. The untokenized text of the second sequence
                      (question).
            endings: list of str. multiple choice's options.
                     Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_examples(self, mode, data_dir, fname, entities_fname):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class JaqketProcessor(DataProcessor):
    """Processor for the Jaqket data set."""

    def _get_entities(self, data_dir, entities_fname):
        logger.info("LOOKING AT {} entities".format(data_dir))
        entities = dict()
        for line in self._read_json_gzip(os.path.join(data_dir, entities_fname)):
            entity = json.loads(line.strip())
            entities[entity["title"]] = entity["text"]

        return entities

    def get_examples(self, mode, data_dir, fname, entities_fname, num_options=20):
        """See base class."""
        logger.info("LOOKING AT {} [{}]".format(data_dir, mode))
        entities = self._get_entities(data_dir, entities_fname)
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fname)),
            mode,
            entities,
            num_options,
        )

    def get_labels(self):
        """See base class."""
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _read_json_gzip(self, input_file):
        with gzip.open(input_file, "rt", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, t_type, entities, num_options):
        """Creates examples for the training and dev sets."""

        examples = []
        skip_examples = 0

        # for line in tqdm.tqdm(
        #    lines, desc="read jaqket data", ascii=True, ncols=80
        # ):
        logger.info("read jaqket data: {}".format(len(lines)))
        for line in lines:
            data_raw = json.loads(line.strip("\n"))

            id = data_raw["qid"]
            question = data_raw["question"].replace("_", "")  # "_" は cloze question
            options = data_raw["answer_candidates"][:num_options]  # TODO
            answer = data_raw["answer_entity"]

            if answer not in options:
                continue

            if len(options) != num_options:
                skip_examples += 1
                continue

            contexts = [entities[options[i]] for i in range(num_options)]
            truth = str(options.index(answer))

            if len(options) == num_options:  # TODO
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=contexts,
                        endings=options,
                        label=truth,
                    )
                )

        if t_type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None

        logger.info("len examples: {}".format(len(examples)))
        logger.info("skip examples: {}".format(skip_examples))

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    logger.info("Convert examples to features")
    features = []
    # for (ex_index, example) in tqdm.tqdm(
    #    enumerate(examples),
    #    desc="convert examples to features",
    #    ascii=True,
    #    ncols=80,
    # ):
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):

            text_a = context
            text_b = example.question + tokenizer.sep_token + ending
            # text_b = tokenizer.sep_token + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation_strategy="only_first",  # 常にcontextをtruncate
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping "
                    "question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = (
                inputs["input_ids"],
                inputs["token_type_ids"],
            )

            # The mask has 1 for real tokens and 0 for padding tokens. Only
            # real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("qid: {}".format(example.example_id))
            for (choice_idx, (input_ids, attention_mask, token_type_ids),) in enumerate(
                choices_features
            ):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info(
                    "attention_mask: {}".format(" ".join(map(str, attention_mask)))
                )
                logger.info(
                    "token_type_ids: {}".format(" ".join(map(str, token_type_ids)))
                )
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features


processors = {"jaqket": JaqketProcessor}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"jaqket", 20}

###############################################################################
###############################################################################

ALL_MODELS = (
    "bert-base-japanese",
    "bert-base-japanese-whole-word-masking",
    "bert-base-japanese-char",
    "bert-base-japanese-char-whole-word-masking",
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultipleChoice, BertJapaneseTokenizer),
}


def select_field(features, field):
    return [
        [choice[field] for choice in feature.choices_features] for feature in features
    ]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
    )
    #########################################################################
    # Check if saved optimizer or scheduler states exist
    t_output_dir = os.path.join(
        args.output_dir, "checkpoint-{}/".format(args.init_global_step)
    )
    if (os.path.isfile(
            os.path.join(t_output_dir, "optimizer.pt")) and
        os.path.isfile(
            os.path.join(t_output_dir, "scheduler.pt")
        )
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(t_output_dir, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(t_output_dir, "scheduler.pt")))
    #########################################################################

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from "
                "https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size,
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) " "= %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    #########################################################################
    # Check if continuing training from a checkpoint
    if os.path.exists(t_output_dir):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            #checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = args.init_global_step  # int(checkpoint_suffix)
            accum_step = args.gradient_accumulation_steps
            t_num_batch = len(train_dataloader) // accum_step
            epochs_trained = global_step // t_num_batch
            steps_trained_in_current_epoch = (global_step * accum_step) % len(train_dataloader)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            logger.info("  %d update per 1 epoch", t_num_batch)
        except ValueError:
            logger.info("  Starting fine-tuning.")
    #########################################################################
    ##################################
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    # best_dev_loss = 99999999999.0
    best_steps = 0
    model.zero_grad()
    train_iterator = tqdm.trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        ascii=True,
        ncols=80,
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:  # epoch
        loss_epoch = 0
        upd_epoch = global_step
        logger.info("Total batch size = %d", len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # training model one step
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)  # forward計算
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # (auto) backward gradient
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # do update (logging and save model)
            t_loss = loss.item()
            tr_loss += t_loss
            loss_epoch += t_loss
            if (step + 1) % args.gradient_accumulation_steps != 0:
                continue
            ################################################
            # update
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if (
                args.local_rank in [-1, 0]
                and args.logging_steps > 0
                and global_step % args.logging_steps == 0
            ):
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss",
                    (tr_loss - logging_loss) / args.logging_steps,
                    global_step,
                )
                logger.info(
                    "Ave.loss: %12.6f Accum.loss %12.6f "
                    "#upd: %5d #iter: %7d lr: %s",
                    (tr_loss - logging_loss) / args.logging_steps,
                    (loss_epoch / max(1, (global_step - upd_epoch))),
                    global_step,
                    step,
                    str(["%.4e" % (lr["lr"]) for lr in optimizer.param_groups]),
                )
                logging_loss = tr_loss

            # save model
            if (
                args.local_rank in [-1, 0]
                and args.save_steps > 0
                and global_step % args.save_steps == 0
            ):
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}".format(global_step)
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_vocabulary(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                torch.save(
                    scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                logger.info(
                    "Saving optimizer and scheduler states to %s", 
                    output_dir)
            # save model END

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=not test, test=test
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm.tqdm(
            eval_dataloader, ascii=True, ncols=80, desc="Evaluating"
        ):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)
        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(
            eval_output_dir, "is_test_" + str(test).lower() + "_output_labels.txt",
        )
        with open(output_eval_file, "w") as fp:
            for pred, out_label_id in zip(preds, out_label_ids):
                fp.write("{} {}\n".format(pred, out_label_id))

        output_eval_file = os.path.join(
            eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt",
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****".format(
                    str(prefix) + " is test:" + str(test)
                )
            )
            writer.write("model           =%s\n" % str(args.model_name_or_path))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = ".".join(args.dev_fname.split(".")[0:-1])
    elif test:
        cached_mode = ".".join(args.test_fname.split(".")[0:-1])
    else:
        cached_mode = ".".join(args.train_fname.split(".")[0:-1])
    assert (evaluate is True and test is True) is False
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    logger.info("Loading features from cached file %s", cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("find %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_examples(
                "dev",
                args.data_dir,
                args.dev_fname,
                args.entities_fname,
                num_options=args.eval_num_options,
            )
        elif test:
            examples = processor.get_examples(
                "test",
                args.data_dir,
                args.test_fname,
                args.entities_fname,
                num_options=args.eval_num_options,
            )
        else:
            examples = processor.get_examples(
                "train",
                args.data_dir,
                args.train_fname,
                args.entities_fname,
                num_options=args.train_num_options,
            )
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=False,
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(
        select_field(features, "input_mask"), dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        select_field(features, "segment_ids"), dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="./data", type=str, help="")
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        choices=(list(MODEL_CLASSES.keys())),
        help=", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-japanese-whole-word-masking",
        type=str,
        help=", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default="jaqket",
        type=str,
        choices=("jaqket"),
        help=", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="")
    parser.add_argument(
        "--train_fname", default="train_questions.json", type=str, help="")
    parser.add_argument(
        "--dev_fname", default="dev1_questions.json", type=str, help="")
    parser.add_argument(
        "--test_fname", default="dev2_questions.json", type=str, help="")
    parser.add_argument(
        "--entities_fname", default="candidate_entities.json", type=str,
        help="")
    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="")
    parser.add_argument("--tokenizer_name", default="", type=str, help="")
    parser.add_argument("--cache_dir", default="", type=str, help="")
    parser.add_argument("--max_seq_length", default=512, type=int, help="")
    parser.add_argument("--do_train", action="store_true", help="")
    parser.add_argument("--do_eval", action="store_true", help="")
    parser.add_argument("--do_test", action="store_true", help="")
    parser.add_argument("--evaluate_during_training", action="store_true", help="")
    # parser.add_argument(
    #     "--do_lower_case", action='store_true', help="")
    parser.add_argument("--train_num_options", default=4, type=int, help="")
    parser.add_argument("--eval_num_options", default=20, type=int, help="")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="")
    parser.add_argument("--max_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=0, type=int, help="")

    parser.add_argument("--logging_steps", type=int, default=50, help="")
    parser.add_argument("--save_steps", type=int, default=50, help="")
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="")
    parser.add_argument("--overwrite_cache", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--fp16", action="store_true", help="")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    # for debugging
    parser.add_argument("--server_ip", type=str, default="", help="")
    parser.add_argument("--server_port", type=str, default="", help="")

    parser.add_argument("--init_global_step", type=int, default=0, help="")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/
        # docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
        "16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        # do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    # initialize_params(model) #########

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        logger.info("### Model training: Start")
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, evaluate=False
        )
        #########################################################################
        t_output_dir = os.path.join(
            args.output_dir, "checkpoint-{}/".format(args.init_global_step)
        )
        if (os.path.isfile(
                os.path.join(t_output_dir, "optimizer.pt")) and
            os.path.isfile(
                os.path.join(t_output_dir, "scheduler.pt")
            )
        ):
            # Load in optimizer and scheduler states
            model = model_class.from_pretrained(t_output_dir)  # , force_download=True)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)
            logger.info(" Loading model from %s [%s]", args.output_dir, t_output_dir)
        #########################################################################
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("### Model training: Done")

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using
        # `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together
        # with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = model_class.from_pretrained(checkpoint)
            # initialize_params(model) #########
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = model_class.from_pretrained(checkpoint)
            # initialize_params(model) #########
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info(
            "best steps of eval acc is the following checkpoints: %s", best_steps,
        )
    return results


if __name__ == "__main__":
    results = main()
    for key, result in results.items():
        print(key, result)
