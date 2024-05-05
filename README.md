## MIHKG-Rec

Author: Zi-Xin Fan(inskybe@gmail.com / inskybe@163.com) Yi-Han Meng

## Introduction

In this work, we propose a novel recommendation algorithm framework that leverages multi-intent user preference representation within heterogeneous knowledge networks.  The framework's working principles are thoroughly described, focusing on enhancing recommender system accuracy through fine-grained user preference representation.

## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We provide three processed datasets: Last FM,Alibaba iFashion,Amazon Book.

see more in `dataloader.py`

## run MIHKG-Rec

Simply use:

```
python run_MIHKG.py --dataset [dataset_name]
```

And the hyperparameters we use are fixed according to the dataset in `MIHKG_Rec.py`.
