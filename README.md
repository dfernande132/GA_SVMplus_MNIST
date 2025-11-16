GAGS: Parallelized Genetic Algorithm-Based Grid Search for SVM+ Model Optimization

ABSTRACT
Support Vector Machines (SVM) have been widely used in supervised learning due to their strong generalization capabilities. SVM+ extends this framework under the Learning Using Privileged Information (LUPI) paradigm, incorporating additional training-stage information unavailable during inference. However, SVM+ involves the tuning of four hyperparameters, doubling the dimensionality of the optimization problem compared to standard SVM, and thereby increasing computational cost substantially.
To address this challenge, this work introduces GAGS (Genetic Algorithm-Based Grid Search), a hybrid GA-based metaheuristic tailored for SVM+ model selection that integrates grid-based structured search, adaptive precision through logarithmic mapping, and advanced parallelization strategies. By distributing computations across independent processing units, GAGS efficiently explores the hyperparameter space, mitigating issues such as premature convergence and local optima entrapment, while drastically reducing execution time.
The proposed GAGS-SVM+ framework is evaluated on a binary classification task using the MNIST dataset under a rigorous protocol of 20 independent runs. Experimental results demonstrate that the GAGS methodology achieves superior accuracy by reducing the forecast error to values below 4.5%, surpassing the best metaheuristic baseline by more than 10% in equal evaluation budget and identical splits across methods. The good performance of GAGS is further confirmed by a notable reduction in test errors under data scarcity scenarios. Furthermore, the parallel implementation of GAGS delivers a performance improvement exceeding ≈4× speed-up on multiprocessor systems, validating its scalability and suitability for high-dimensional, resource-intensive kernel machine optimization tasks.

REPOSITORY FILES 

FOM_LUPI.m
    Fitness function for SVM+ (LUPI). Computes the dual/objective value and metrics in both spaces (FX/FX*, FR). Called by the runner scripts.

GAGS.m
    Main implementation of GAGS (Genetic Algorithm–based Grid Search) to optimize (C, γ, σ, σ*). Includes log-scale initialization, genetic operators, and sub-grid parallelization.

README.md
    Top-level documentation. Should include how to run, dependencies, and how to reproduce the paper’s tables/figures.

cwru_12kDE_1hp_X_Xstar_y.mat
    .mat dataset CWRU bearing with X (decision), X* (privileged), and y (labels).

predict_LUPI.m
    Prediction routine for SVM+. Given a trained solution (parameters and biases), returns predictions in the decision and/or correction space.

results_GAGS_B4000_S20_n100.mat
    Aggregated results for GAGS (budget B=4000, S=20 runs, n_train=100). Contains best parameters, metrics (FX/FX*, FR), runtimes, and seeds.

results_BO_B4000_S20_n100.mat
    Aggregated results for the Bayesian Optimization baseline under the same protocol (B=4000, S=20, n=100).

results_PSO_B4000_S20_n100.mat
    Aggregated results for the Particle Swarm Optimization baseline (same protocol).

results_DE_B4000_S20_n100.mat
    Aggregated results for the Differential Evolution baseline (same protocol).

results_RS_B4000_S20_n100.mat
    Aggregated results for the Random Search baseline (same protocol).

run_GAGS.m
    Orchestrates GAGS: sets limits/ranges, depth, budget, parallelization; calls FOM_LUPI and saves results_GAGS_*.mat.

run_BO.m
    Runner for Bayesian Optimization: defines optimizable variables, acquisition, and budget; saves results_BO_*.mat.

run_PSO.m
    Runner for PSO with the same bounds and budget; saves results_PSO_*.mat.

run_DE.m
    Runner for Differential Evolution (e.g., rand/1/bin) with the same bounds and budget; saves results_DE_*.mat.

run_RS.m
    Runner for Random Search (uniform or log-uniform sampling as configured) with the common budget; saves results_RS_*.mat.

svmPlus_Model.m
    SVM+ model implementation/structure (kernel computation, dual solver, recovery of b and b*). Used by FOM_LUPI/predict_LUPI.

test_reducted.mat
    .mat dataset MNIST  with X (decision), X* (privileged), and y (labels) for test.

train_reducted.mat
    .mat dataset MNIST  with X (decision), X* (privileged), and y (labels) for train.

-------------------------------------------------------------------
Suggested reproduction order:
1) Check README.md for environment and versions (MATLAB R2024b, required toolboxes).
2) Ensure *train_reducted.mat* and *test_reducted.mat* are present with expected variables.
3) Run baselines: run_RS.m, run_BO.m, run_PSO.m, run_DE.m (all with B=4000, S=20, n_train=100).
4) Run run_GAGS.m with the same budget and seeds.
5) Load the results_*.mat files to build the paper’s tables/figures.
6) (Optional) Use cwru_12kDE_1hp_X_Xstar_y.mat as an CWRU additional dataset.
