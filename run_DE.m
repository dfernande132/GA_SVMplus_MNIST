function results = run_DE(data, B, S, ntrain, seed_subsets, seed_opt_base, NP, F, CR)
% RUN_DE - Differential Evolution (rand/1/bin) en espacio lineal con proxy y post-procesado lexicográfico
%
% Uso:
%   results = run_DE('train_reducted.mat', 5000, 30, 200);
%
% Notas:
% - FOM_LUPI devuelve: [valPlus, ~, ~, ~, rr], con rr = [numSucc, numSuccCorr, tot, accX, accX*]
%   => obj = valPlus; accX = rr(4); accX* = rr(5).
% - Selección para el algoritmo: minimizar el PROXY consistente con 𝓛(p).
% - Reporte final: post-proceso lexicográfico estricto sobre TODAS las evaluaciones registradas.
% - rand/1/bin con reparación por clipping a los límites lineales.

    if nargin < 2 || isempty(B), B = 5000; end
    if nargin < 3 || isempty(S), S = 30;   end
    if nargin < 4 || isempty(ntrain), ntrain = 200; end
    if nargin < 5 || isempty(seed_subsets), seed_subsets = 12345; end
    if nargin < 6 || isempty(seed_opt_base), seed_opt_base = 54321; end
    if nargin < 7 || isempty(NP), NP = min(50, max(20, floor(B/50))); end  % tamaño de población
    if nargin < 8 || isempty(F),  F  = 0.7; end
    if nargin < 9 || isempty(CR), CR = 0.9; end

    % --- Cargar datos ---
    fvAll     = data.X_train;
    lblAll    = data.y_train;
    fvStarAll = data.PI_train;

    N = size(fvAll,1);
    assert(N >= S*ntrain, 'Muestras insuficientes: %d < %d.', N, S*ntrain);

    % --- Cotas lineales ---
    lb = [1e-3, 1e-4, 1e-2, 1e-2];
    ub = [1e+3, 1e+2, 1e+2, 1e+2];
    D  = numel(lb);

    % --- Subconjuntos disjuntos reproducibles ---
    rng(seed_subsets,'twister');
    perm    = randperm(N);
    idx_all = reshape(perm(1:S*ntrain), ntrain, S);

    % --- Resultados por subset ---
    results(S,1) = struct('subset_id',[],'seed_subsets',[],'seed_opt',[],'idx_subset',[], ...
                          'best_params',[],'best_obj',[],'accX',[],'accXstar',[], ...
                          'calls',[],'time_seconds',[],'best_result_row',[]);

    % --- Presupuesto: evals ≈ NP (init) + NP * G  ->  G = floor((B - NP)/NP)
    maxGenerations = max(0, floor((B - NP) / max(1,NP)));

    % --- Pesos del proxy (jerarquía accX >> accX* >> obj) ---
    eps1 = 1e-3; eps2 = 1e-6;

    for s = 1:S
        idx    = idx_all(:,s);
        fv     = fvAll(idx,:);
        lbl    = lblAll(idx,:);
        fvStar = fvStarAll(idx,:);

        rng(seed_opt_base + s,'twister');

        % --- Trazas de evaluaciones (todas las llamadas a FOM_LUPI) ---
        traceStd   = NaN(B,7);  % [C,gamma,sigma,sigmaStar, obj, accX, accX*]
        evalIdx    = 0;

        % --- Inicializar población (uniforme en espacio lineal) ---
        Pop = lb + rand(NP, D) .* (ub - lb);
        Fit = inf(NP,1);

        % Evaluar población inicial (cuenta para B)
        for i = 1:NP
            if evalIdx >= B, break; end
            [Fit(i), rowStd] = evalCandidate(Pop(i,:));
            evalIdx = evalIdx + 1; traceStd(evalIdx,:) = rowStd;
        end
        if evalIdx >= B
            % Post-procesado y salida del subset
            [best_row, calls] = postprocess(traceStd, evalIdx);
            results = writeSubset(results, s, seed_subsets, seed_opt_base, idx, best_row, calls, 0);
            continue;
        end

        % --- Evolución por generaciones ---
        t0 = tic;
        for g = 1:maxGenerations
            for i = 1:NP
                if evalIdx >= B, break; end

                % --- rand/1 mutation: v = a + F*(b - c), con a,b,c distintos de i
                [a,b,c] = distinctABC(NP, i);
                v = Pop(a,:) + F * (Pop(b,:) - Pop(c,:));

                % --- binomial crossover ---
                u = Pop(i,:);
                jr = randi(D);  % asegúrate de copiar al menos una dimensión
                for j = 1:D
                    if rand <= CR || j == jr
                        u(j) = v(j);
                    end
                end

                % --- Reparación por clipping a [lb,ub] ---
                u = max(min(u, ub), lb);

                % --- Evaluar trial y selección por proxy ---
                [fitU, rowStd] = evalCandidate(u);
                evalIdx = evalIdx + 1; traceStd(evalIdx,:) = rowStd;

                if fitU <= Fit(i)
                    Pop(i,:) = u;
                    Fit(i)   = fitU;
                end
            end
            if evalIdx >= B, break; end
        %fprintf('Generacion numero %d de %d para S=%d\n',g,maxGenerations,S);
        end
        elapsed = toc(t0);

        % --- Post-procesado lexicográfico y volcado de resultados ---
        [best_row, calls] = postprocess(traceStd, evalIdx);
        results = writeSubset(results, s, seed_subsets, seed_opt_base, idx, best_row, calls, elapsed);

        fprintf('[DE Subset %2d/%2d] accX=%.4f | accX*=%.4f | obj=%.6g | evals=%d | time=%.1fs\n', ...
            s, S, results(s).accX, results(s).accXstar, results(s).best_obj, results(s).calls, results(s).time_seconds);
    end

    outname = sprintf('results_DE_B%d_S%d_n%d.mat', B, S, ntrain);
    save(outname,'results');
    fprintf('Resultados DE guardados en %s\n', outname);

    % ======= helpers internos =======

    function [y, stdRow] = evalCandidate(x)
        % x = [C gamma sigma sigmaStar]
        C = x(1); gamma = x(2); sigma = x(3); sigmaStar = x(4);
        [valPlus, ~, ~, ~, rr] = FOM_LUPI(fv, fvStar, lbl, C, gamma, sigma, sigmaStar);
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
        % Proxy a minimizar (jerarquía fuerte):
        aX = accX;  if isnan(aX),  aX  = 0; end
        aS = accXs; if isnan(aS), aS = 0; end
        sp = log1p(exp(-abs(obj))) + max(obj,0);  % softplus estable
        y  = -(aX) + 1e-3*(1 - aS) + 1e-6*log1p(sp);

        stdRow = [C, gamma, sigma, sigmaStar, obj, accX, accXs];
    end

    function [best_row, calls] = postprocess(traceStd, evalIdx)
        calls = evalIdx;
        R = traceStd(1:evalIdx, :);
        % Filtrar filas no evaluadas (NaN en obj)
        valid = ~isnan(R(:,5));
        R = R(valid, :);
        if isempty(R)
            best_row = NaN(1,7); return;
        end
        accX  = R(:,6);  accXs = R(:,7);  objv = R(:,5);
        accX_rank  = accX;  accX_rank( isnan(accX_rank) )  = -inf;
        accXs_rank = accXs; accXs_rank( isnan(accXs_rank) ) = -inf;
        obj_rank   = objv;  obj_rank( isnan(obj_rank) )     =  inf;

        maxAccX = max(accX_rank); I  = find(accX_rank == maxAccX);
        maxAccXs = max(accXs_rank(I)); I2 = I(accXs_rank(I) == maxAccXs);
        [~,krel] = min(obj_rank(I2)); k = I2(krel);

        best_row = R(k,:);
    end

    function results = writeSubset(results, s, seed_subsets, seed_opt_base, idx, best_row, calls, elapsed)
        results(s).subset_id       = s;
        results(s).seed_subsets    = seed_subsets;
        results(s).seed_opt        = seed_opt_base + s;
        results(s).idx_subset      = idx(:)';
        results(s).best_params     = best_row(1:4);
        results(s).best_obj        = best_row(5);
        results(s).accX            = best_row(6);
        results(s).accXstar        = best_row(7);
        results(s).best_result_row = best_row;
        results(s).calls           = calls;
        results(s).time_seconds    = elapsed;
    end

    function [a,b,c] = distinctABC(NP, i)
        % elige 3 índices distintos y distintos de i
        idx = 1:NP; idx(i) = [];
        p = idx(randperm(NP-1,3));
        a = p(1); b = p(2); c = p(3);
    end
end
