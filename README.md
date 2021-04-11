# Warehouse Storage Optimisation

Optimizing your warehouse means examining every corner of your infrastructure and every facet of your workflows and processes to identify and correct inefficiencies. Not only does warehouse optimization result in a healthier bottom line, but it also improves key warehouse metrics like accurate orders and on-time delivery.

![360_F_274776871_8DH1CBKK3MAAD3kS21aClKyhl3xQKgYJ](https://user-images.githubusercontent.com/46780667/114293818-ed3f3500-9ab6-11eb-88f9-8e2ff7037fe0.jpg)

Our solution for this optimization problem attempts to use Machine Learning techniques to provide a scalable and fast solution to this problem. We used several models like Decision tree, Random Forest Classifier, XGBoost classifier, k-Nearest Neighbours, etc to solve this issue.

# Results
The highest accuracy we obtained so far is 89% in Random Forest Classifier. The following graph shows our confusion matrix plot for RFC at various values of n_estimators 
### Confusion Matrix
<img src="https://github.com/nisarg14/CSE523-Machine-Learning-Gophers/blob/main/Results/confusion_matrix.png" alt="confusion_matrix" width="900" height="600"/>

As you can see in the graph, the classification of item into -1th bin is where our error lies. When the bins are full and the algorithm is supposed to not place the item, our algorithm is placing it into a bin. That is where the algorithm is malperforming,

## Final result of Random Forest Classifier with varying depth and number of estimators

<img src="https://github.com/nisarg14/CSE523-Machine-Learning-Gophers/blob/main/Results/rfc_hyperparameters_1.png" alt="rfc_hyperparameter_1" width="800" height="300"/>

<img src="https://github.com/nisarg14/CSE523-Machine-Learning-Gophers/blob/main/Results/rfc_hyperparameters_2.png" alt="rfc_hyperparameter_2" width="800" height="300"/>

# Find our Dataset at

1. [Amazon Bin Image Dataset](https://www.kaggle.com/dhruvildave/amazon-bin-image-dataset)
2. [Tabularised Amazon Bin Image Dataset](https://www.kaggle.com/dhatrikapuriya/tabularised-dataset-of-amazon-bin-dataset)

3. [CTGAN Sythetic Data](https://www.kaggle.com/dhatrikapuriya/merged-data)



# References

1. [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
2. [Optimizing Warehouse Operations with Machine Learning on GPUs, C. Seward](https://developer.nvidia.com/blog/optimizing-warehouse-operations-machine-learning-gpus/)
3. [Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.](https://optuna.org/#paper)
4. [Erin LeDell and Sebastien Poirier. H2O AutoML: Scalable Automatic Machine Learning. 7th ICML Workshop on Automated Machine Learning (AutoML), July 2020](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_61.pdf)
5. https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

6. https://mlli.mit.edu/projects/machine-learning/predicting-warehouse-storage

7. https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/

8. https://developer.nvidia.com/blog/optimizing-warehouse-operations-machine-learning-gpus/

9. https://en.wikipedia.org/wiki/Multiclass_classification

10. https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

11. https://optuna.org/

12. https://automl.github.io/auto-sklearn/master/

