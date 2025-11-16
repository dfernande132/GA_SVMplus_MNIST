function results = run_PSO(data, B, S, ntrain, seed_subsets, seed_opt_base, swarmSize)
% PARTICLESWARMOPTIMIZATION - PSO en espacio lineal con proxy y post-procesado lexicográfico.
%
% Uso:
%   results = ParticleSwarmOptimization('train_reducted.mat', 5000, 30, 200);
%
% Notas:
% - FOM_LUPI devuelve: [valPlus, ~, ~, ~, rr], con rr = [numSucc, numSuccCorr, tot, accX, accX*]
%   => obj = valPlus; accX = rr(4); accX* = rr(5).
% - Para el reporte usamos la regla 𝓛(p): max accX → max accX* → min obj.
% - El presupuesto B se respeta diseñando SwarmSize y MaxIterations.

    if nargin < 2 || isempty(B), B = 5000; end
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

    % --- Tamaño del enjambre y nº iteraciones para respetar B ---
    if nargin < 7 || isempty(swarmSize)
        % Heurística: enjambre moderado, pero ≤ B
        swarmSize = min(50, max(10, floor(B/20)));
    end
    % nº de evaluaciones ~ SwarmSize (init) + SwarmSize*MaxIterations
    maxIterations = max(0, floor((B - swarmSize) / max(1,swarmSize)));

    for s = 1:S
        idx    = idx_all(:,s);
        fv     = fvAll(idx,:);
        lbl    = lblAll(idx,:);
        fvStar = fvStarAll(idx,:);

        rng(seed_opt_base + s,'twister');

        % --- Trazas: guardamos cada evaluación en formato estándar ---
        traceStd   = NaN(B,7);   % [C, gamma, sigma, sigmaStar, obj, accX, accX*]
        traceValid = false(B,1);
        global evalIdx;
        evalIdx = 0;

        % --- Objetivo (proxy consistente con 𝓛), sin efectos laterales ---
        eps1 = 1e-3; eps2 = 1e-6;
        objectivePSO = @(x) objProxyAndLog(x, fv, fvStar, lbl, eps1, eps2);

        % --- PSO options ---
        % Semilla reproducible y enjambre inicial dentro de [lb,ub]
        initSwarm = lb + rand(swarmSize, D) .* (ub - lb);
        options = optimoptions('particleswarm', ...
            'SwarmSize',          swarmSize, ...
            'MaxIterations',      maxIterations, ...
            'Display',            'off', ...
            'UseVectorized',      false, ...
            'InitialSwarmMatrix', initSwarm, ...
            'HybridFcn',          []);  % sin híbrido

        % --- Ejecutar PSO (con límites) ---
        t0 = tic;
        try
            particleswarm(objectivePSO, D, lb, ub, options);
        catch ME
            warning('PSO error en subset %d: %s', s, ME.message);
        end
        elapsed = toc(t0);

        % --- Post-procesado lexicográfico estricto ---
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

            maxAccX = max(accX_rank);
            I  = find(accX_rank == maxAccX);
            maxAccXs = max(accXs_rank(I));
            I2 = I(accXs_rank(I) == maxAccXs);
            [~,krel] = min(obj_rank(I2));
            k = I2(krel);

            best_row = R(k,:);
        end

        % --- Volcar resultados ---
        results(s).subset_id       = s;
        results(s).seed_subsets    = seed_subsets;
        results(s).seed_opt        = seed_opt_base + s;
        results(s).idx_subset      = idx(:)';
        results(s).best_params     = best_row(1:4);
        results(s).best_obj        = best_row(5);
        results(s).accX            = best_row(6);
        results(s).accXstar        = best_row(7);
        results(s).best_result_row = best_row;
        results(s).calls           = min(evalIdx, B);  % por si PSO llamó menos
        results(s).time_seconds    = elapsed;

        fprintf('[PSO Subset %2d/%2d] accX=%.4f | accX*=%.4f | obj=%.6g | evals=%d | time=%.1fs\n', ...
            s, S, results(s).accX, results(s).accXstar, results(s).best_obj, results(s).calls, elapsed);
    end

    outname = sprintf('results_PSO_B%d_S%d_n%d.mat', B, S, ntrain);
    save(outname,'results');
    fprintf('Resultados PSO guardados en %s\n', outname);


    % ===== Objetivo PSO: calcula proxy y registra evaluación =====
    function y = objProxyAndLog(x, fv, fvStar, lbl, eps1, eps2)
        % x es un vector [C gamma sigma sigmaStar] en espacio lineal
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

        % Proxy (minimizar): -accX  >>  -accX*  >>  obj
        aX  = accX;  if isnan(aX),  aX  = 0; end
        aXs = accXs; if isnan(aXs), aXs = 0; end
        sp  = log1p(exp(-abs(obj))) + max(obj,0);  % softplus estable
        y   = -(aX) + eps1*(1 - aXs) + eps2*log1p(sp);

        % Registrar en formato estándar
        if evalIdx < B
            evalIdx = evalIdx + 1;
            traceStd(evalIdx,:) = [C, gamma, sigma, sigmaStar, obj, accX, accXs];
            traceValid(evalIdx) = true;
        end
        %fprintf('evaluacion numero  %d\n', evalIdx);
    end
end
