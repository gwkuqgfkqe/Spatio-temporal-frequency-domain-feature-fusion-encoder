config: Used to define hyperparameters for each model during training
dataset_loader: reads data and preprocesses data
experiments: Training code
nn_models: STFEnc and other comparison models
save: Saves training logs and results
trained_model: Some trained models
main.py: Training master file
SVM.py: The accuracy rate is obtained by classifying the encoded data
tsne.py: Reduce and cluster the encoded data and visualize them separately

The code runs as follows:
First, run the main file to train and save the model, then run the SVM file to classify and calculate the accuracy of the encoded eeg data, and finally run the tsne file to reduce and cluster the encoded eeg data and do visualization
