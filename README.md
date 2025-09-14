# deep_learning_lyrics_generation

This repository contains the implementation of Assignment 3 in the Deep Learning course at Ben-Gurion University.  
The project implements Recurrent Neural Networks (RNNs with LSTM) for lyrics generation conditioned on melodies.  
The dataset includes MIDI files (melodies) and song lyrics, and the task is to generate realistic lyrics aligned with the given melody.

The assignment focuses on:
- Building an LSTM-based sequence-to-sequence model for text generation  
- Integrating melody information from MIDI files into the lyrics model  
- Comparing two approaches for melody integration:
  - Method 1: Binary instrument presence vectors (128-dim)  
  - Method 2: Rich musical features (tempo, key, intervals, dynamic range)  
- Experimenting with hyperparameters (learning rate, batch size, hidden size)  
- Analyzing generated lyrics conditioned on different starting words  

---

## Project Structure

- `lyrics_generation_method1.ipynb` – Jupyter Notebook implementing Method 1 (binary instrument vectors)  
- `lyrics_generation_method2.ipynb` – Jupyter Notebook implementing Method 2 (rich musical features)  
- `lyrics_generation_report.pdf` – Report summarizing preprocessing, methodology, experiments, and results  

---

## Dataset

- Lyrics: CSV files containing lyrics of 600 training songs and 5 test songs  
- Melodies: MIDI files, analyzed with PrettyMIDI  
- Preprocessing steps included:
  - Cleaning lyrics (removing punctuation, annotations, contractions)  
  - Tokenization with NLTK and regex  
  - Word embeddings using pre-trained Word2Vec Google News 300d vectors  
  - Mapping lyrics to corresponding MIDI files for alignment  
  - Creating padded sequences (max length = 50 tokens)  

---

## Model Architectures

- Embedding Layer: Converts words into 300-dim vectors (Word2Vec pretrained)  
- Melody Integration:  
  - Method 1: Concatenates word embeddings with a 128-dim instrument vector  
  - Method 2: Concatenates word embeddings with 4 rich features (tempo, key, intervals, dynamic range)  
- LSTM Layer: Hidden size = 256, captures sequential dependencies  
- Fully Connected Layer: Outputs vocabulary-sized probability distribution for next word prediction  
- Loss Function: CrossEntropyLoss (ignoring padding tokens)  
- Sampling: Next word chosen by sampling from probabilities, not argmax  

---

## Training Setup

- Train/Validation split: 90% / 10%  
- Batch sizes tested: 16, 32, 64  
- Learning rates tested: 0.01, 0.001, 0.0001, 0.00001  
- Optimal configuration: LR = 0.001, Batch Size = 16  
- Epochs: up to 30, with early stopping (patience = 5)  
- Framework: PyTorch, with TensorBoard logging  

---

## Results

### Method 1 – Instrument Vector  
- Best Validation Loss: 5.6459 (Epoch 11, early stopping at 16)  
- Average Test Loss: 5.7709 across 5 test songs  

### Method 2 – Rich Features  
- Best Validation Loss: 5.6580 (Epoch 10, early stopping at 15)  
- Average Test Loss: 5.9330 across 5 test songs  

Both approaches performed consistently, with Method 2 showing slightly better generalization thanks to richer melody representation.  

Generated lyrics were evaluated using different initial words (love, dream, party), showing strong influence of the seed word on the thematic direction of the song.  

---

## Setup

Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn pretty_midi nltk gensim
```
Make sure to also download NLTK tokenizers if not already installed:
```bash
import nltk
nltk.download('punkt')
```

---

## Run
1. Open one of the notebooks: lyrics_generation_method1.ipynb, lyrics_generation_method2.ipynb
2. Run all cells to preprocess data, train the model, and generate lyrics.
3. Training and validation curves will be logged.
4. Generated lyrics for test songs will be saved in the notebook outputs.

