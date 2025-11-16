% Predict_LUPI
%
% Function created by Jose Daniel Fernandez - v1 Decembre 2024
% based FOM_LUPI(Features Optimized Model for LUPI)
% 
% The predict function takes the parameters of the model and the vectors 
% to be classified, and returns the classification based on SVM+.

function [result] = predict_LUPI(modelo, data)

    npoints = size(data, 1);
    numSVPlus = size(modelo.solPlus.alphaPlus, 1);
    idxSVPlus = find(modelo.solPlus.alphaPlus >= 0);

    % ****** CALCULATE PREDICTION ********
    predictionPlus = modelo.bPlus*ones(npoints, 1);
    for i1 = 1:npoints % Se hace la prediccion de cada punto de la descarga desde los vectores soporte 
        for i2 = 1:numSVPlus
            predictionPlus(i1) = predictionPlus(i1)  + modelo.lbl(idxSVPlus(i2))*modelo.solPlus.alphaPlus(idxSVPlus(i2))*SvkernelRFB(modelo.fv(idxSVPlus(i2), :), data(i1, :), modelo.sgm);
        end
    end
result=predictionPlus;
end

function k = SvkernelRFB(u, v, p)
    k = exp(-pdist2(u, v).^2 / (2 * p^2)); 
end
