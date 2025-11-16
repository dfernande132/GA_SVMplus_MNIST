% FOM_LUPI
%
% Function created by Jose Daniel Fernandez - v1 October 2024
% FOM_LUPI(Features Optimized Model for LUPI)
% 
% The Lagrangians alpha and beta are obtained (the SVM+ model is created)
% Check restrictions of (ec.23) Vapnik article
% Input the feature vectors fv and fvStar (privileged information).
% Input the labels (+1/-1) are in the vector lbl
% Input the four parameters SVM+ model
% The function returns the fvalPlus value, the Lagrangians to build the model
% ( alpha y betha), and the vector result with num Succes,
% num Succes Correction, total feature vectors, percentage Succes and
% percentage Succes Correction;

function [valPlus, solPlus, bPlus, bStar, result] = FOM_LUPI(fv, fvStar, lbl, Cparam, gammaParam, sgmPlus, sgmStar)

    Threshold = 1e-8; 
    zeroAprox = 1e-6; 
    numfv = size(fv, 1);
    idxSV = (1:numfv)';
    idxDisruptiveFV = find(lbl < 0);
    idxNonDisruptiveFV = find(lbl > 0);

    yLab = lbl * lbl';
    KMatrPlus =zeros(numfv,numfv); % prelocated 
    KMatrStar =zeros(numfv,numfv); % prelocated 
    for i1 = 1:numfv
        for i2 = 1:numfv
            KMatrPlus(i1, i2) = exp(-(fv(i1, :)-fv(i2, :))*(fv(i1, :)-fv(i2, :))'/(2*sgmPlus^2));
            KMatrStar(i1, i2) = exp(-(fvStar(i1, :)-fvStar(i2, :))*(fvStar(i1, :)-fvStar(i2, :))'/(2*sgmStar^2));
        end
    end
    
    % ******** OPTIMIZATION PROBLEM ********
    alphaPlus = optimvar('alphaPlus', numfv, 'LowerBound', 0);
    betaPlus = optimvar('betaPlus', numfv, 'LowerBound', 0);

    % Precomputed variables
    precomputedKMatrPlus = yLab .* KMatrPlus;
    precomputedgamma = 0.5 / gammaParam;
    % Solve definition
    probPlus = optimproblem;
    probPlus.Objective = -sum(alphaPlus) + 0.5 * sum(sum((alphaPlus * alphaPlus') .* precomputedKMatrPlus)) + ...    
    (precomputedgamma) * sum(sum(((alphaPlus + betaPlus - Cparam) * (alphaPlus + betaPlus - Cparam)').* KMatrStar));
    
    probPlus.Constraints.cons1 = alphaPlus' * lbl == 0;
    probPlus.Constraints.cons2 = sum(alphaPlus + betaPlus - Cparam) == 0;

    optPlus = optimoptions(probPlus, 'MaxIterations', 800, 'Display', 'off');
    [solPlus, valPlus, ~, ~, ~] = solve(probPlus, 'Options', optPlus);

    % Setting values below Threshold to zero
    solPlus.alphaPlus(solPlus.alphaPlus < Threshold) = 0;
    solPlus.betaPlus(solPlus.betaPlus < Threshold) = 0;

    % ****** CALCULATE THE BIAS ********
    bparamPlus = zeros(numfv, 1);
    bparamPlusStar = zeros(numfv, 1);

    for i = 1:numfv
        bparamPlus(i) = lbl(idxSV(i)) - (lbl(idxSV).*solPlus.alphaPlus(idxSV))'*KMatrPlus(idxSV, idxSV(i));
        bparamPlusStar(i) = lbl(idxSV(i)) - (1/gammaParam)*((solPlus.alphaPlus(idxSV) + solPlus.betaPlus(idxSV) - Cparam)'*KMatrStar(idxSV, idxSV(i)));
    end
    bPlus = mean(bparamPlus);
    bStar = mean(bparamPlusStar);

    % ****** CALCULATE THE CLASSIFICATION PERCENTAGE ********
    % Decision space predictions
    predTrainingM1 = (lbl(idxSV) .* solPlus.alphaPlus(idxSV))' * SvkernelRFB(fv(idxSV, :), fv(idxDisruptiveFV, :), sgmPlus) + bPlus;
    predTrainingP1 = (lbl(idxSV) .* solPlus.alphaPlus(idxSV))' * SvkernelRFB(fv(idxSV, :), fv(idxNonDisruptiveFV, :), sgmPlus) + bPlus;
    
    % Correction space predictions
    predTrainingCorrM1 = ((solPlus.alphaPlus(idxSV) + solPlus.betaPlus(idxSV) - Cparam)' * SvkernelRFB(fvStar(idxSV, :), fvStar(idxDisruptiveFV, :), sgmStar)) / gammaParam + bStar;
    predTrainingCorrP1 = ((solPlus.alphaPlus(idxSV) + solPlus.betaPlus(idxSV) - Cparam)' * SvkernelRFB(fvStar(idxSV, :), fvStar(idxNonDisruptiveFV, :), sgmStar)) / gammaParam + bStar;

    % Calculate success of predictions
    tot = numel(idxDisruptiveFV) + numel(idxNonDisruptiveFV);
    numSuccM1 = sum(predTrainingM1 < 0);
    numSuccP1 = sum(predTrainingP1 > 0);
    numSucc = numSuccM1 + numSuccP1;
    numSuccM1corr = sum(predTrainingCorrM1 < 0);
    numSuccP1corr = sum(predTrainingCorrP1 > 0);
    numSuccCorr = numSuccM1corr + numSuccP1corr;
    result = [numSucc, numSuccCorr, tot, numSucc / tot, numSuccCorr / tot];

    % ****** CHECK RESTRICTIONS AND RETURN RESULT ******
    A=solPlus.alphaPlus; 
    B=solPlus.betaPlus; 
    sumaTotal = 0;
    iszero=true;
    
    for i = 1:size(A, 1)
        sumaFila = A(i) + B(i) - Cparam;   
        if abs(sumaFila) > zeroAprox
            iszero=false;
        end    
        sumaTotal = sumaTotal+sumaFila;
    end
    
    if iszero || abs(sumaTotal)>zeroAprox  % Does not meet the constraints
        solPlus.alphaPlus=0;
        solPlus.betaPlus=0;
        valPlus=0;
        bPlus=0;
        bStar=0;
        result=[0,0,0,0,0];
    end
end

function k = SvkernelRFB(u, v, p)
    k = exp(-pdist2(u, v).^2 / (2 * p^2)); 
end
