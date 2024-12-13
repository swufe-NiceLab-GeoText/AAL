# Adversity-aware Few-shot Named Entity Recognition via Augmentation Learning

ALL is a Variational Prototype-Augmented Learning model. It designed to retrieve and reinforce entity prototypes resilient to adversarial inference, thereby enhancing cross-domain semantic coherence.


## Requirement
	numpy  
	torch==1.10.0  
	transformers==4.42.3  
	scikit-learn==0.24.2  
	pot==0.7.0  
	seqeval

##  Datasets
For snips datasets and cross-domain datasets, you can download the dataset from [here](https://atmahou.github.io/attachments/ACL2020data.zip).

For the attack dataset, you can do the following for the snips dataset and the cross-domain dataset.
The textual adversarial attack algorithm BERT-Attack (Li et al. 2020) is used to perform synonym substitution and generate adversarial examples. The codes of BERT-Attack can be found at [https://github.com/LinyangLee/BERT-Attack](https://github.com/LinyangLee/BERT-Attack).

After processing the dataset put it in the same directory as the original dataset and add attack to the file name to differentiate it, for example `xval_attack_ner`.
##  Training and Evaluation
We provide examples of training and evaluating models in the `scripts` folder. For example, to train and evaluate a model on the News dataset in a 1-shot setup, simply run the following command:

	bash scripts/Domain1_1shot.sh
 
