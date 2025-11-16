GAGS: Parallelized Genetic Algorithm-Based Grid Search for SVM+ Model Optimization

ABSTRACT
Support Vector Machines (SVM) have been widely used in supervised learning due to their strong generalization capabilities. SVM+ extends this framework under the Learning Using Privileged Information (LUPI) paradigm, incorporating additional training-stage information unavailable during inference. However, SVM+ involves the tuning of four hyperparameters, doubling the dimensionality of the optimization problem compared to standard SVM, and thereby increasing computational cost substantially.
To address this challenge, this work introduces GAGS (Genetic Algorithm-Based Grid Search), a hybrid GA-based metaheuristic tailored for SVM+ model selection that integrates grid-based structured search, adaptive precision through logarithmic mapping, and advanced parallelization strategies. By distributing computations across independent processing units, GAGS efficiently explores the hyperparameter space, mitigating issues such as premature convergence and local optima entrapment, while drastically reducing execution time.
The proposed GAGS-SVM+ framework is evaluated on a binary classification task using the MNIST dataset under a rigorous protocol of 20 independent runs. Experimental results demonstrate that the GAGS methodology achieves superior accuracy by reducing the forecast error to values below 4.5%, surpassing the best metaheuristic baseline by more than 10% in equal evaluation budget and identical splits across methods. The good performance of GAGS is further confirmed by a notable reduction in test errors under data scarcity scenarios. Furthermore, the parallel implementation of GAGS delivers a performance improvement exceeding ≈4× speed-up on multiprocessor systems, validating its scalability and suitability for high-dimensional, resource-intensive kernel machine optimization tasks.

FILES IN REPOSITORY

train_binary.mat  % 200 samples of MNIST black and white

train200.mat      % 200 samples of MNIST with Privileged information

test_reducted.mat  % 200 samples of MNIST

svmplusModelGA.mat  % Optimized SVM+ Model

svmPlusModel.mat % Non-optimized SVM+ Model

svmModel.mat % SVM Model

