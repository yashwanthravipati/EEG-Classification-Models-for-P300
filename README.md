# Evaluating Deep Learning performance of P300 Speller BCI Signal Classifcation  

This code is created to compare the P300 EEG Signal classification performance of traditional methods such as LDA with novel deep learning approaches EEGNet and EEGInception.

1. Across-Subject Training and Testing.ipynb
 This code will train EEG-Net, RG+xDAWN and rLDA algorithms to classify P300 neural signal with 15 subjects data batched together. Also, inclded the code to test and save results to a CSV with 60 subject files.
 
2. Across-Subject-EEG-Inception.ipynb
  This code will train EEGInception algorithm to classify P300 neural signal with 15 subjects data batched together. Also, inclded the code to test and save results to a CSV with 60 subject files.
  
3. Within-Subject-P300-Classification.ipynb
  This code will train EEG-Net, EEG-Inception, rLDA and RG+xDAWN algorithms for each subject (with sessions data batched together, 80%/20% train/test split) and test the algorithm effectiveness with test data. 
  
 
