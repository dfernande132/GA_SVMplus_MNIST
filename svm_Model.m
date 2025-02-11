%% Script para cargar los datos de entrenamiento y crear un modelo SVM
% Se asume que el archivo train.mat contiene la estructura 'train' con:
%   - X_train: matriz de características de tamaño [n x 100]
%   - y_train: vector de etiquetas de tamaño [n x 1] (con valores -1 y 1)
%   - PI_train: información privilegiada (no se usa en este SVM)
%
% El modelo se entrena utilizando X_train y y_train, y se guarda en svmModel.mat

% 1. Cargar los datos de entrenamiento
load('train200.mat');   % Carga la estructura 'train'

% Verificar el tamaño de los datos
fprintf('Datos de entrenamiento: %d muestras, %d características\n', size(train.X_train,1), size(train.X_train,2));

% 2. Entrenar el modelo SVM
% Se utiliza fitcsvm para clasificación binaria, con kernel rbf y
% estandarización de las características.

svmModel = fitcsvm(train.X_train, train.y_train, ...
    'KernelFunction', 'linear', 'Standardize', true);

% % 3. (Opcional) Realizar validación cruzada para estimar el desempeño
% cvSVMModel = crossval(svmModel);
% cvLoss = kfoldLoss(cvSVMModel);
% fprintf('Pérdida de validación cruzada: %.4f\n', cvLoss);

% Predecir las etiquetas en el conjunto de entrenamiento
trainPred = predict(svmModel, train.X_train);
% Calcular el número de muestras clasificadas correctamente
numCorrect = sum(trainPred == train.y_train);
totalSamples = size(train.X_train, 1);

fprintf('Muestras correctas en modelo: %d de %d.\n', numCorrect, totalSamples);

% 4. Guardar el modelo entrenado
save('svmModel.mat', 'svmModel');
fprintf('Modelo SVM guardado en svmModel.mat\n');
