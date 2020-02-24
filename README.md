# JAQKET_baseline


```
mkdir ../data
cd .../data
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json
wget https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz
gunzip candidate_entities.json.gz

mkdir ../working_dir
cd ../working_dir

cp ../JAQKET_baseline/run_jaqket_baseline_sample.sh .

./run_jaqket_baseline_sample.sh outputs
```

