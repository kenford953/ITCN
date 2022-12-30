# Contextual Relation Embedding and Interpretable Triplet Capsule for Inductive Relation Prediction
Code for ITCN (Interpretable Triplet Capsule Network), Neurocomputing 2022, https://www.sciencedirect.com/science/article/abs/pii/S0925231222008992

This repository includes data, code and pretrained models for the Neurocomputing 2022 paper, " Contextual Relation Embedding and Interpretable Triplet Capsule for Inductive Relation Prediction".

## Run the Code
### Requirements
- Python 3.7
- Pytorch 1.5.0
- CUDA 10.2
The requirements are packed in the file "environment.yml". You can create the conda invironment by 
~~~
conda env create -f environment.yml
~~~

### Train and test
Train the model with grid search
~~~~
python main_refined_grid_search.py --dataset [the name of dataset]
~~~~
The argument "--dataset" could be [nell-v1, nell-v2, nell-v3, nell-v4, FB15k-237-v1, FB15k-237-v2, FB15k-237-v3, FB15k-237-v4]. The reported results and the corresponding hyperparameters will be presented in the terminal. All the trained models are saved in the "./pretrained_model/".
We save the pretrained models in the './best_pretrained_model/', which could be used to achieve the reported results in our paper.
~~~
python main.py --dataset [the name of dataset] --train False --test True --pretrained_model [the filename of the pretrained model]
~~~


