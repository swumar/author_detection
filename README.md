# Spooky Author Identification

Author Identification is a very important and practical problem in Natural Language Processing. Every author has his/her own unique style of writing their literary work. These writing styles of authors are recognizable by human readers. The process of identifying the authors based on their writing skills from a collection of different authors is known as Author Identification. This process has become well known because of its widespread applications where the author of the document is unknown and in detecting plagiarized articles, in recommendation systems where a reader is recommended with other authors based on the writing style of his favourite author and in detecting the spoofed emails

## Problem Statement:

There are many literary works in the past wherein the authors of these works are unknown. Our aim is to identify the authors of these works by building a model, using NLP techniques, which learns from the writing style of various different authors. For our project we have particularly considered the horror stories by authors like Edgar Allan Poe, Mary Shelley and HP Lovecraft.

For the problem statement mentioned, we will be using different feature vectors like TF-IDF, LDA and encodings and attention masks from BERT followed by different classifiers like Logistic Regression, SVM and BERT Sequence classifier, for better results.

## Results:

**Tf-idf with Logistic Regression:**<br> 
f1_score for EAP,HPL,MWS is 0.81436077 0.79338843 0.80341113
Confusion matrix for this model:
<p align="center">
  <img width="500" height="400" src= results/tf_idf.png>
</p>


**BERT encodings with BERT classifier:**<br>
f1_score for EAP,HPL,MWS is 0.89251018 0.90884956 0.88198236<br>
Confusion matrix for this model:
<p align="center">
  <img width="475" height="375" src= results/BERT.png>
</p>
