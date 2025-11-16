function results = run_GAGS(data, B, S, ntrain, seed_subsets, seed_opt_base)
% RUN_GAGS - Wrapper para ejecutar GAGS en S subconjuntos y guardar resultados homogéneos
%
% Uso:
%   results = run_GAGS('train_reducted.mat', 5000, 30, 200);
%
% Requisitos:
%   - datafile contiene variable 'train' con campos:
%       train.X_train, train.y_train, train.PI_train
%   - GAGS en el path con firma:
%       [bestResult, llam] = GAGS(fv, fvStar, lbl, depth, B)
%     donde bestResult = [C, gamma, sigma, sigmaStar, obj, accX, accXstar]
%           llam       = nº de llamadas a FOM_LUPI realizadas por GAGS
%
% Notas:
%   - depth se fija a 1 como pides.
%   - Se generan S subconjuntos DISJUNTOS de tamaño ntrain, reproducibles con seed_subsets.
%   - Se fija rng por subconjunto para reproducibilidad de GAGS (si usa aleatoriedad interna).
%   - Se guarda 'GAGS_results_B<B>_S<S>_n<ntrain>.mat'.

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
    assert(N >= S*ntrain, 'No hay suficientes muestras (%d) para %d subconjuntos de %d.', N, S, ntrain);

    % --- Subconjuntos disjuntos reproducibles ---
    rng(seed_subsets, 'twister');
    perm = randperm(N);
    idx_all = reshape(perm(1:S*ntrain), ntrain, S);

    % --- Estructura de resultados homogénea ---
    results(S,1) = struct('subset_id',[],'seed_subsets',[],'seed_opt',[],'idx_subset',[], ...
                          'best_params',[],'best_obj',[],'accX',[],'accXstar',[], ...
                          'calls',[],'time_seconds',[],'best_result_row',[]);

    depth = 1;

    % --- Bucle por subconjuntos ---
    for s = 1:S
        idx    = idx_all(:,s);
        fv     = fvAll(idx,:);
        lbl    = lblAll(idx,:);
        fvStar = fvStarAll(idx,:);

        % Semilla por subset para reproducibilidad (si GAGS usa RNG interno)
        seed_opt = seed_opt_base + s;
        rng(seed_opt, 'twister');

        t0 = tic;
        [bestResult, llam] = GAGS(fv, fvStar, lbl, depth, B);

        elapsed = toc(t0);

        % Normaliza bestResult a 1x7 por si acaso
        bestResult = bestResult(:).';
        if numel(bestResult) < 7
            bestResult = [bestResult, nan(1, 7-numel(bestResult))];
        elseif numel(bestResult) > 7
            bestResult = bestResult(1:7);
        end

        % Volcar resultados
        results(s).subset_id       = s;
        results(s).seed_subsets    = seed_subsets;
        results(s).seed_opt        = seed_opt;
        results(s).idx_subset      = idx(:)';
        results(s).best_params     = bestResult(1:4);
        results(s).best_obj        = bestResult(5);
        results(s).accX            = bestResult(6);
        results(s).accXstar        = bestResult(7);
        results(s).best_result_row = bestResult;
        results(s).calls           = llam;       % nº real de llamadas que informa GAGS
        results(s).time_seconds    = elapsed;

        fprintf('[GAGS Subset %2d/%2d] accX=%.4f | accX*=%.4f | obj=%.6g | calls=%d | time=%.1fs\n', ...
            s, S, results(s).accX, results(s).accXstar, results(s).best_obj, results(s).calls, results(s).time_seconds);
        
        % Aviso si GAGS no respetó el presupuesto
        if ~isempty(B) && llam > B
            warning('GAGS Subset %d: GAGS realizó %d llamadas (> B=%d).', s, llam, B);
        end
    end

    % Guardado homogéneo
    outname = sprintf('results_GAGS_B%d_S%d_n%d.mat', B, S, ntrain);
    save(outname, 'results');
    fprintf('Resultados GAGS guardados en %s\n', outname);
end
