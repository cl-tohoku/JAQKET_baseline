# JAQKET_baseline

__Home__:
https://www.nlp.ecei.tohoku.ac.jp/projects/aio/

__LeaderBoard__:
https://www.nlp.ecei.tohoku.ac.jp/projects/AIP-LB/task/aio

## Requirements
python 3.6 or later

Several python liblaries
```
$ pip install tqdm
$ pip install numpy
$ pip install torch
$ pip install tensorboard
$ pip install transformers
$ pip install mecab-python3
```


## Sample
```
$ mkdir ../data
$ cd ../data
$ wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json
$ wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json
$ wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json
$ wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz

$ mkdir ../working_dir
$ cd ../working_dir

$ cp ../JAQKET_baseline/run_jaqket_baseline_sample.sh .

$ ./run_jaqket_baseline_sample.sh
```

### Launch
You can run these setting commands using `./scripts/preprocess.sh`


## submit
After training, you would convert a prediction file into a submission format.

```code
$ cd [your working directory]
$ python ../JAQKET_baseline/scripts/create_submission_file.py \
    -test ../data/aio_leaderboard.json \
    -pred output_dir/is_test_true_output_labels.txt \
    -fo submission.json \
    -wq -wc
```

And that, try to submit `submission.json` from the following page: https://www.nlp.ecei.tohoku.ac.jp/projects/AIP-LB/task/aio
