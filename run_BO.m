function results = run_BO(data, B, S, ntrain, seed_subsets, seed_opt_base)
% BO en espacio lineal con proxy y post-procesado lexicográfico estricto.
% FOM_LUPI devuelve:
%   [valPlus, ~, ~, ~, rr],  rr = [numSucc, numSuccCorr, tot, accX, accX*]
% -> obj = valPlus; accX = rr(4); accX* = rr(5).

    if nargin < 2 || isempty(B), B = 1000; end
    if nargin < 3 || isempty(S), S = 30;   end
    if nargin < 4 || isempty(ntrain), ntrain = 200; end
    if nargin < 5 || isempty(seed_subsets), seed_subsets = 12345; end
    if nargin < 6 || isempty(seed_opt_base), seed_opt_base = 54321; end

    % --- Cargar datos ---
    fvAll     = data.X_train;
    lblAll    = data.y_train;
    fvStarAll = data.PI_train;

    N = size(fvAll,1);
    assert(N >= S*ntrain, 'Muestras insuficientes: %d < %d.', N, S*ntrain);

    rng(seed_subsets,'twister');
    perm    = randperm(N);
    idx_all = reshape(perm(1:S*ntrain), ntrain, S);

    % Variables (espacio LINEAL, sin transformaciones)
    vars = [ ...
        optimizableVariable('C',[1e-3,1e3],'Transform','none'), ...
        optimizableVariable('gamma',[1e-4,1e2],'Transform','none'), ...
        optimizableVariable('sigma',[1e-2,1e2],'Transform','none'), ...
        optimizableVariable('sigmaStar',[1e-2,1e2],'Transform','none')];

    % Salida
    results(S,1) = struct('subset_id',[],'seed_subsets',[],'seed_opt',[],'idx_subset',[], ...
                          'best_params',[],'best_obj',[],'accX',[],'accXstar',[], ...
                          'calls',[],'time_seconds',[],'best_result_row',[]);

    for s = 1:S
        idx    = idx_all(:,s);
        fv     = fvAll(idx,:);
        lbl    = lblAll(idx,:);
        fvStar = fvStarAll(idx,:);

        rng(seed_opt_base + s,'twister');

        % Trazas normalizadas por evaluación: [C,gamma,sigma,sigma*, obj, accX, accX*]
        traceStd   = NaN(B,7);
        traceValid = false(B,1);
        global evalIdx;
        evalIdx = 0;

        % Proxy con jerarquía (guía al BO, no decide el reporte)
        eps1 = 1e-3; eps2 = 1e-6;

        objectiveBO = @(x) objProxyAndLog(x,fv,fvStar,lbl,eps1,eps2);

        t0 = tic;
        bayesopt(objectiveBO, vars, ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'MaxObjectiveEvaluations', B, ...
            'IsObjectiveDeterministic', false, ...
            'Verbose', 0, 'PlotFcn', {}, 'UseParallel', false);
        elapsed = toc(t0);

        % --- Selección lexicográfica estricta sobre el trace ---
        K = find(traceValid);
        if isempty(K)
            warning('Subset %d: ninguna evaluación válida registrada.', s);
            best_row = NaN(1,7);
        else
            R = traceStd(K,:);
            accX  = R(:,6);
            accXs = R(:,7);
            objv  = R(:,5);

            accX_rank  = accX;  accX_rank( isnan(accX_rank) )  = -inf;
            accXs_rank = accXs; accXs_rank( isnan(accXs_rank) ) = -inf;
            obj_rank   = objv;  obj_rank( isnan(obj_rank) )     =  inf;

            maxAccX = max(accX_rank); I  = find(accX_rank == maxAccX);
            maxAccXs = max(accXs_rank(I)); I2 = I(accXs_rank(I) == maxAccXs);
            [~,krel] = min(obj_rank(I2));
            k = I2(krel);

            best_row = R(k,:);
        end

        results(s).subset_id       = s;
        results(s).seed_subsets    = seed_subsets;
        results(s).seed_opt        = seed_opt_base + s;
        results(s).idx_subset      = idx(:)';
        results(s).best_params     = best_row(1:4);
        results(s).best_obj        = best_row(5);
        results(s).accX            = best_row(6);
        results(s).accXstar        = best_row(7);
        results(s).best_result_row = best_row;
        results(s).calls           = B;
        results(s).time_seconds    = elapsed;

        fprintf('[BO Subset %2d/%2d] accX=%.4f | accX*=%.4f | obj=%.6g | time=%.1fs\n', ...
            s, S, results(s).accX, results(s).accXstar, results(s).best_obj, results(s).time_seconds);
    end

    outname = sprintf('results_BO_B%d_S%d_n%d.mat', B, S, ntrain);
    save(outname,'results');
    fprintf('Resultados BO guardados en %s\n', outname);

    % ===== Objetivo: calcular proxy y registrar evaluación =====
    function y = objProxyAndLog(x,fv,fvStar,lbl,eps1,eps2)
        [valPlus, ~, ~, ~, rr] = FOM_LUPI(fv, fvStar, lbl, x.C, x.gamma, x.sigma, x.sigmaStar);
        % rr = [numSucc, numSuccCorr, tot, accX, accX*]
        obj  = valPlus;
        accX = NaN; accXs = NaN;
        if ~isempty(rr)
            rr = rr(:).';
            if numel(rr) >= 5
                accX  = rr(4);
                accXs = rr(5);
            end
        end

        % Registro estandarizado
        stdRow = [x.C, x.gamma, x.sigma, x.sigmaStar, obj, accX, accXs];

        % Proxy (minimizar): -accX  >>  -accX*  >>  obj
        aX  = accX;  if isnan(aX),  aX  = 0; end
        aXs = accXs; if isnan(aXs), aXs = 0; end
        sp  = log1p(exp(-abs(obj))) + max(obj,0);  % softplus estable
        y   = -(aX) + eps1*(1 - aXs) + eps2*log1p(sp);

        if evalIdx < B
            evalIdx = evalIdx + 1;
            traceStd(evalIdx,:) = stdRow;
            traceValid(evalIdx) = true;
        end
        % fprintf('evaluacion numero  %d\n', evalIdx);
    end
end
