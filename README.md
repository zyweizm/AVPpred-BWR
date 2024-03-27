# AVPpred_RBW
Prediction of Antiviral Peptides

The peptides possessing potential to inhibit the virus are considered as antiviral peptides (AVPs). Usually, the AVPs exhibit antiviral effects by inhibiting the virus directly. With the development of molecular biology techniques, some data-driven tools have emerged to predict antiviral peptides. However, it is necessary to improve the predictive performance of these tools for antiviral peptides.

In this study, we propose a biological words model for the first time and create a model named AVPpred_RBW designed for the prediction of antiviral peptides. To the best of our knowledge, this is the first time that realized the word segmentation of protein primary structure sequence based on the regularity of protein secondary structure.
Finally, our model achieves 99.1% AUC, 93.83% MCC, 96.9%,98.3% SN, 95.4% SP on the training dataset. compared with the state-of-the-art model, AUC increases by 4.7%, MCC increases by 11.2% and SP increases by 8.9%. Meanwhile, we generalize the model and tests on the Anticancer activity dataset and the DPP IV inhibitory activity dataset, which still achieves better performance than the existing model.
