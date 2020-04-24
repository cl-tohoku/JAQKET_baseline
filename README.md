# JAQKET_baseline

## Requirements
python 3.6 or later

Several python liblaries
```
pip install tqdm
pip install numpy
pip install torch
pip install tensorboard
pip install transformers
pip install mecab-python3
```


## Sample
```
mkdir ../data
cd ../data
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz

mkdir ../working_dir
cd ../working_dir

cp ../JAQKET_baseline/run_jaqket_baseline_sample.sh .

./run_jaqket_baseline_sample.sh
```

