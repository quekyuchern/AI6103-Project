# AI6103 Deep Learning Project

## 20 Newsgroup Dataset

### Overview
The **20 Newsgroup Dataset** is a collection of approximately 18,000 newsgroup documents, partitioned across 20 different newsgroups. It is widely used for text classification and clustering tasks, making it an ideal dataset for exploring deep learning architectures. Each document belongs to one of the 20 categories, ranging from technology to politics and sports.

### Project Structure
This project investigates various deep learning architectures for text classification using the 20 Newsgroup Dataset. Below is an overview of the folders and the experiments they contain:

#### **Attention**
This folder contains experiments utilizing attention mechanisms to improve the model's ability to focus on important words or phrases in the text. These mechanisms help capture contextual information effectively for text classification.

#### **Baseline**
The baseline models serve as a reference point for comparison. These include simple models such as Logistic Regression or Multilayer Perceptrons (MLPs) without advanced architectural components like CNNs or LSTMs.

#### **CNN**
This folder contains experiments that explore Convolutional Neural Networks (CNNs) for text classification. The models focus on extracting local features and patterns from the text data using convolutional layers.

#### **Embeddings**
The embeddings folder includes experiments that test various word embedding techniques such as Word2Vec, GloVe, and pre-trained embeddings like BERT. The goal is to analyze the impact of different embeddings on model performance.

#### **ExploreHyperParameters**
This folder contains hyperparameter tuning experiments. It explores various combinations of learning rates, optimizers, dropout rates, and batch sizes to identify optimal configurations for the deep learning models.

#### **LSTM**
The Long Short-Term Memory (LSTM) folder contains experiments that leverage recurrent neural networks to capture sequential dependencies in the text data. It focuses on the temporal aspects of the dataset.

#### **hybrid_cnn_lstm**
This folder includes experiments combining CNN and LSTM architectures to leverage the strengths of both. CNNs extract local features, while LSTMs capture sequential dependencies, creating a robust hybrid model.

In addition, I have included our originial brainstorming document, Brainstorm.ipynb.
