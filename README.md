# Optimizing an ML Pipeline in Azure

## Overview
This project is a component of the Udacity Azure ML Nanodegree. It entails the construction and refinement of an Azure ML pipeline utilizing the Python SDK and a provided Scikit-learn model. The ensuing model is then evaluated against an Azure AutoML run.

## Summary
The project leverages data stemming from the direct marketing endeavors of a banking institution in Portugal, predominantly conducted via phone calls. The dataset encompasses 20 variables, including age, occupation, and marital status, with the goal column delineating two categories—Yes and No—to signify whether the client subscribed to a term deposit at the bank.

The algorithms developed through the Python SDK (with Hyperdrive) and AutoML strive to precisely predict the likelihood of a potential client subscribing to a term deposit. This facilitates optimal resource allocation towards clients with a higher propensity to subscribe.

A Voting Ensemble model discovered through the AutoML run exhibited the highest accuracy at 91.77%. However, the Logistic classifier, shaped using Hyperdrive, closely followed with an accuracy of 90.59%.

## Scikit-learn and Hyperdrive Pipeline

### Scikit-learn
Initially, a Logistic Regression model was formulated and trained employing Scikit-learn in the train.py. The process delineated in the Python script encompasses:

- Incorporation of the banking dataset via Azure TabularDataset Factory
- Application of a cleaning function for data purification and transformation
- Division of processed data into training and testing subsets
- Utilization of Scikit-learn for the preliminary training of a Logistic Regression model, specifying the values of C and max_iter. These parameters were intended for subsequent optimization via Hyperdrive.
- Preservation of the trained model

With parameters C=3122262070046.714 and max_iter=500, the model attained an accuracy of 90.59%.

### Hyper Drive
The initially trained model underwent optimization with Hyperdrive, allowing for efficient, parallel hyperparameter tuning. The Hyperdrive implementation consisted of:

- Azure cloud resources configuration
- Hyperdrive configuration
- Execution of the Hyperdrive
- Retrieval of the optimal model

**RandomParameterSampling** and **BanditPolicy** were pivotal in the Hyperdrive configuration, enabling efficient exploration of hyperparameter spaces and early termination of underperforming runs.

## AutoML

### Best Model Description:

The best model generated by AutoML is a Voting Ensemble model. A Voting Ensemble (or majority voting ensemble) is a model that combines the predictions from multiple machine learning algorithms. It is often used to achieve higher accuracy and model stability. This is especially beneficial in situations where different models perform well for different types of input data, as the ensemble can leverage the strengths of each constituent model.

### Parameters of the Best Model:

#### 1. `voting`: 'soft'
This parameter implies that the ensemble model is using Soft Voting. It takes the weighted average of probabilities for class membership predicted by the individual models and then predicts the class with the highest average probability.

#### 2. `weights`: [0.35714285714285715, 0.07142857142857142, 0.07142857142857142, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142]
This parameter represents the weights assigned to the constituent models in the ensemble. A higher weight indicates that the associated model’s vote has more influence on the final prediction. In this instance, the first model in the ensemble has the most influence on the final output.

#### 3. `n_jobs`: None
This parameter is typically used to specify the number of jobs to run in parallel for fit, predict, etc. `None` usually means 1, i.e., the computations will not run in parallel.

#### 4. `flatten_transform`: None
This parameter is usually related to whether the transformation should be flattened or not. `None` in this case possibly implies that the default behavior is used, and no flattening transformation is applied.

You can review the full list of parametres in [params](params.txt)

## Pipeline Comparison
AutoML holds a slight edge due to its streamlined architecture and superior accuracy. The essential advantage of AutoML over Hyperdrive lies in AutoML's capacity to effortlessly evaluate diverse algorithms. This is evident in this project, where a model untested by us outperformed our selected model.

## Future work

Progress can be achieved by refining the Voting Ensemble algorithm from AutoML using Hyperdrive for hyperparameter tuning. Hyperdrive can explore varied parameter sampling methods, possibly unveiling a hyperparameter set that surpasses the accuracy attained by AutoML.

Given the bank's focus on accurately pinpointing willing subscribers and avoiding resource wastage on unwilling clients, a model's precision takes precedence over its recall. Hence, applying a beta of 0.5 will accentuate precision, necessitating the utilization of the F-beta score for evaluating both the Hyperdrive and AutoML models.
