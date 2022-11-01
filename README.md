
# MetaVerse Mind Lab Take-Home Assessment
This repository contains the submission for the take-home assessment for mv.ml. This is a text classification task where the input is news article text with their headings and author information. 

## Problem Solution Approach
Various solutions can be possible for a simple text classification problem. In this case, the efficiency matters as well as the performance of the algorithms. This, I tried various Machine Learning Algorithms with Word2Vec for Embeddings and an LSTM model as well to get the best performance for this task in given time. 
It turns out that in this case, just a normal machine learning algorithm with good feature engineering outperforms even the LSTM model.

 ## Implementation Details
Since using the entire text information as input would increase the time and space complexity by more than 50 times, I tried to use an aggregate of the `author` and article `titles` as features. First these texts are passed though a data cleaning pipeline, where the text is converted to lowercase and then the punctuations are removed. Finally, the stopwords are also removed and the text is then ready for generating word embeddings. 

The next step is to use Word2Vec to generate the word embeddings. Word2Vec uses the Continuous Bag of Words (CBOW) model to generate the embeddings. After the embeddings are generated, usually all the sentences are padded with zeroes to ensure that the length of every sentence is uniform. But, to represent an entire document, it would take a lot of space and consequently, time as well. So, I took element-wise mean of all the vectors of the words in a document. This would mean that every single document will be represented by an embedding of limited length and all of them would be uniform. 

These embeddings are then used to pass them through Classifiers like Logistic Regression, Random Forest, Adaboost and  K Nearest Neighbors. 

## Evaluation 
I tried 5 different models for evaluation of this project. They are as follows: 
	1. Word2Vec + Logistic Regression 
	2. Word2Vec + Random Forest Classifier
	3. Word2Vec + Adaboost Classifier 
	4. Word2Vec + K Nearest Neighbors 
	5. Tf-iDF + LSTM
For their evaluation, I used the `F1 Score`. Since it is very important to identify the fake news articles, in our case the the True Positives are of the most importance for us. Thus, it is important to balance the precision and recall. 

The results of the evaluation are as follows: 

| Model| Validation F1 - Score | Testing F1 - Score  |
| ----------- | ----------- |  ----------- |  
| Word2Vec + Logistic Regression| 0.87|0.68|
| Word2Vec + Random Forest Classifier| 0.92 |0.73 |
| Word2Vec + Adaboost Classifier| 0.95 |0.65 |
| Word2Vec + K Nearest Neighbors| 0.96 |0.69 |
| Tf-iDF + LSTM| 0.88 |0.61 |

From the above table, it is evident that the Machine Learning models give a better score. This is the reason why I have included all the ML models in the `models.py` file. However, I have also uploaded an `assessment.ipynb` that has all the experiments with ML models and LSTMs. 

## Future Scope of Improvement 
The Machine Learning models can be improved with further feature engineering like using a dependency tree, or adding additional information using Named Entity Recognition (NER). There can also be potential improvements if we use the entire text instead of just the title and author (however, this would require a lot of space, time and hardware resources). 
We can also use autoencoders to generate the embeddings for further improvement and fine-tuning. 
