# Project 1. Sentiment Analysis with BIRT

## Overview
In this project, we will use **Hugging Face Transformers** and pre-trained **BERT** Neural Networks for sentiment analysis. We will run the model using a single prompt but also leverage **BeautifulSoup** to scrape reviews from Yelp to be able to calculate sentiment on a larger scale.

There are three main steps that we are going to follow:
1. Download and install BERT from HF Transformers
2. Run sentiment analysis Using BERT and Python
3. Scrape reviews from Yelp and calculate the score



# Project 2. Fine-Tune BERT for Text Classification with TensorFlow

## Overview
In this project, we will use TensorFlow and TF-Hub to fine-tune a BERT model (BIRT: Bidirectional Embedding Representations from Transformers) for text classification. To train such a model, we mainly have to train the classifier, with minimal changes happening to the BERT model during the training phase. This training process is called Fine-Tuning, and has roots in Semi-supervised Sequence Learning and ULMFiT (https://jalammar.github.io/illustrated-bert/). 
We use a pre-trained BERT model that is [available](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2) on [TensorFlow Hub](https://tfhub.dev/). 

The objective of this project is to:
- Build TensorFlow Input Pipelines for Text Data with the [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) API
- Tokenize and Preprocess Text for BERT
- Fine-tune BERT for text classification with TensorFlow 2 and [TF Hub](https://tfhub.dev)

For this project, we will be using a [data set](https://www.kaggle.com/c/quora-insincere-questions-classification) provided y **Quora** on  **Kaggle** to detect toxic online content to improve the quality of discussions and conversations we have online. **Quora**  is one of the well-known question and answering websites that provided this dataset as a competition on Kaggle. The data is labeled and contains questions asked on Quora and corresponding labels that indicate whether the question was insincere or not. Insincere questions are considered to be toxic, these questions can have a non-neutral tone, could be aggressive or disparaging, or inflammatory or based on false information, or containing sexual content for shock value and so on. If the question is insincere, it is labeled as 1, and a non-toxic question is labeled as 0.

There are some preprocessing steps that are required to get the raw input data to a format that BIRT accepts as input. However, luckily for us, the official TensorFlow models repositories has sub modules and tokenizers that do these preprocessing for us. So, we are not going to do any additional preprocessing manually, like lower casing, stemming, removing stop words and so on. Because, the module that we will be using from TensorFlow Hub along with the submodules and helper functions from the official model repository is going to take care of this for us.
