%% Script para cargar los datos de entrenamiento y crear un modelo SVM+
% Se asume que el archivo train.mat contiene la estructura 'train' con:
%   - X_train: matriz de características de tamaño [n x 100]
%   - y_train: vector de etiquetas de tamaño [n x 1] (con valores -1 y 1)
%   - PI_train: información privilegiada (no se usa en este SVM)
%
% El modelo se guarda en svmPlusModel.mat

% selecciono parametros

parametros=[3.286418000000000e+03,0.051900000000000,16.311600000000000,72.962300000000000];

Cparam      = parametros(1) ;
gammaParam  = parametros(2);
sgm         = parametros(3);
sgmStar     = parametros(4);

ntrain=125;
dataset=2; % 1->MINST; 2->CWRU
S=1;

if dataset==1
    datafile = 'train_reducted.mat';
    data = load(datafile);
    data=data.train;
else
    datafile = 'CWRU_Dataset/cwru_12kDE_1hp_X_Xstar_y.mat';
    Sim = load(datafile);
    X     = double(Sim.X);
    Xstar = double(Sim.Xstar);
    y     = double(Sim.y(:));
    data = struct();
    data.X_train  = X;    
    data.PI_train = Xstar;   
    data.y_train  = y;   
end

fvAll     = data.X_train;
lblAll    = data.y_train;
fvStarAll = data.PI_train;
N = size(fvAll, 1);
seed_subsets = 12345;
seed_opt_base = 54321;
rng(seed_subsets, 'twister');
perm = randperm(N);
idx_all = reshape(perm(1:S*ntrain), ntrain, S);  % cada columna = un subconjunto
idx = idx_all(:, 1);
fv     = fvAll(idx, :);
lbl    = lblAll(idx, :);
fvStar = fvStarAll(idx, :);


[valPlus, solPlus, bPlus, bStar, result] = FOM_LUPI(fv, fvStar, lbl, Cparam, gammaParam, sgm, sgmStar);
%[valPlus2, solPlus2, bPlus2, bStar2, result2] = QFOM_LUPI(fv, fvStar, lbl, Cparam, gammaParam, sgm, sgmStar);

svmplusModel.bPlus=bPlus;
svmplusModel.fv=fv;
svmplusModel.fvStar=fvStar;
svmplusModel.lbl=lbl;
svmplusModel.solPlus=solPlus;
svmplusModel.sgm=sgm;
if valPlus==0
    fprintf('No cumple las restricciones: %d\n', valPlus);
else
    fprintf('valor de ValPlus: %d\n', valPlus);
    fprintf('Muestras correctas en corrección: %d\n', result(1));
    fprintf('Muestras correctas en decision: %d\n', result(2));
    save('svmplusModel.mat', 'svmplusModel');
    fprintf('Modelo guardado en svmplusModel.mat\n'); 
end

