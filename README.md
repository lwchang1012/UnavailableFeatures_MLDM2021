# MFL

## Reference

> Learning to Predict With Unavailable Features: an End-to-End Approach via Knowledge Transferring, Chun-Chen Lin, Li-Wei Chang, Chun-Pai Yang, Shou-De Lin, MLDM 2021

See the "docs" directory in this repo for the main body and supplementary PDF files.

## Environment (recommended)

- Python 3.6.9
- tensorflow 2.4.1
  - CUDA 11.2
  - cuDNN 8
- pandas 1.1.5
- scikit-learn 0.24.1

## Run Example

- Run proposed training algorithm on the data set "LETTER" 10 times and print average performance

```shell
python3 decay_distill.py \
  --dataset letter --miss-type random --miss-ratio 0.2 \
  --num-exp 10 --epochs 700 --batch-size 128 \
  --dropout-mode random-drop --dropout-begin-step 0.0 --dropout-end-step 20000.0 \
  --temp 1.0 --lamb 0.8 --lr 0.01
```

## To-Do

- Support all data sets reported in the paper
- Code architecture & style refactoring
- Tutorial for applying this code on other data sets
