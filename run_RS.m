% Random Search
function results = run_RS(data, B, S, ntrain, seed_subsets, seed_opt_base)
% RUN_RANDOMSEARCH - Búsqueda aleatoria en espacio lineal con selección lexicográfica
%
% Uso:
%   results = run_RandomSearch('train_reducted.mat', 5000, 30, 200);
%
% Parámetros:
%   datafile       - ruta al .mat que contiene 'train' con campos:
%                    train.X_train, train.y_train, train.PI_train
%   B              - presupuesto de evaluaciones por subconjunto (p.ej., 5000)
%   S              - número de subconjuntos (p.ej., 30)
%   ntrain         - tamaño de cada subconjunto (p.ej., 200)
%   seed_subsets   - (opcional) semilla para crear subconjuntos únicos
%   seed_opt_base  - (opcional) base de semilla para el optimizador por subconjunto
%
% Salida (struct array results de longitud S):
%   .subset_id         - índice del subconjunto (1..S)
%   .seed_subsets      - semilla usada para particionar
%   .seed_opt          - semilla usada para muestrear parámetros en este subconjunto
%   .idx_subset        - índices (en el dataset completo) usados en este subconjunto
%   .best_params       - [C, gamma, sigma, sigmaStar] del mejor según 𝓛(p)
%   .best_obj          - valor de la función objetivo (columna 5 de result)
%   .accX              - precisión en X (columna 6 de result)
%   .accXstar          - precisión en X* (columna 7 de result)
%   .calls             - número de llamadas (igual a B)
%   .time_seconds      - tiempo total de la búsqueda en este subconjunto
%   .best_result_row   - fila completa devuelta por FOM_LUPI (1x7)
%
% Requisitos:
%   - FOM_LUPI debe estar en el path: 
%     [valPlus, solPlus, bPlus, bStar, result] = ...
%        FOM_LUPI(fv, fvStar, lbl, Cparam, gammaParam, sgm, sgmStar);
%     donde 'result' = [C, gamma, sigma, sigmaStar, obj, accX, accXstar]
%
% Nota:
%   - Muestreo UNIFORME en espacio LINEAL dentro de los límites:
%       C in [1e-3, 1e3]
%       gamma in [1e-4, 1e2]
%       sigma in [1e-2, 1e2]
%       sigmaStar in [1e-2, 1e2]
%   - Selección lexicográfica: max accX, luego max accX*, luego min obj.
%   - Subconjuntos disjuntos, reproducibles.

    if nargin < 2 || isempty(B), B = 5000; end
    if nargin < 3 || isempty(S), S = 30;   end
    if nargin < 4 || isempty(ntrain), ntrain = 200; end
    if nargin < 5 || isempty(seed_subsets), seed_subsets = 12345; end
    if nargin < 6 || isempty(seed_opt_base), seed_opt_base = 54321; end

    % --- Cargar datos ---
    fvAll     = data.X_train;
    lblAll    = data.y_train;
    fvStarAll = data.PI_train;

    N = size(fvAll, 1);
    if N < S*ntrain
        error('No hay suficientes muestras (%d) para %d subconjuntos disjuntos de %d.', N, S, ntrain);
    end

    % --- Definir límites lineales ---
    lb = [1e-3, 1e-4, 1e-2, 1e-2];
    ub = [1e+3, 1e+2, 1e+2, 1e+2];

    % --- Subconjuntos disjuntos reproducibles ---
    rng(seed_subsets, 'twister');
    perm = randperm(N);
    idx_all = reshape(perm(1:S*ntrain), ntrain, S);  % cada columna = un subconjunto

    % --- Preparar resultados ---
    results(S,1) = struct('subset_id',[],'seed_subsets',[],'seed_opt',[],'idx_subset',[], ...
                          'best_params',[],'best_obj',[],'accX',[],'accXstar',[], ...
                          'calls',[],'time_seconds',[]);

    % --- Bucle por subconjuntos ---
    for s = 1:S
        bestFitness=[0,0,0,0,0,0,0];
        idx = idx_all(:, s);
        fv     = fvAll(idx, :);
        lbl    = lblAll(idx, :);
        fvStar = fvStarAll(idx, :);

        % Semilla del optimizador para este subconjunto (reproducible y diferente)
        seed_opt = seed_opt_base + s;
        rng(seed_opt, 'twister');

        t0 = tic;
        for b = 1:B
            % Muestreo uniforme en espacio LINEAL
            params = lb + rand(1,4) .* (ub - lb);
            Cparam      = params(1);
            gammaParam  = params(2);
            sgm         = params(3);
            sgmStar     = params(4);

            % Llamada a la función de fitness (contabiliza una llamada aunque falle)
                [valPlus, ~, ~, ~, resultRow] = FOM_LUPI(fv, fvStar, lbl, Cparam, gammaParam, sgm, sgmStar);
                accX  = resultRow(4);
                accXs = resultRow(5);
                fitness=[Cparam,gammaParam,sgm,sgmStar,valPlus,accX,accXs];
                % Regla lexicográfica 𝓛(p): max accX, luego max accX*, luego min obj
                if isBetterLexi([accX, accXs, -valPlus], [bestFitness(6),bestFitness(7),-bestFitness(5)])
                    bestFitness    = fitness;
                end

        end
        elapsed = toc(t0);

        % Guardar resultados del subconjunto
        results(s).subset_id       = s;
        results(s).seed_subsets    = seed_subsets;
        results(s).seed_opt        = seed_opt;
        results(s).idx_subset      = idx(:)';
        results(s).best_params     = bestFitness(1:4);
        results(s).best_obj        = bestFitness(5);
        results(s).accX            = bestFitness(6);
        results(s).accXstar        = bestFitness(7);
        results(s).calls           = B;
        results(s).time_seconds    = elapsed;

        % (Opcional) imprimir un pequeño resumen por subconjunto
        fprintf('[RS Subset %2d/%2d] accX=%.4f | accX*=%.4f | obj=%.6g | time=%.1fs\n', ...
            s, S, bestFitness(6), bestFitness(7), bestFitness(5), elapsed);
    end

    % Guardado automático
    outname = sprintf('results_RS_B%d_S%d_n%d.mat', B, S, ntrain);
    save(outname, 'results');
    fprintf('Resultados guardados en %s\n', outname);
end

 
% ---------- helpers ----------
function tf = isBetterLexi(a, b)
% Devuelve true si a es mejor que b según la prioridad:
% a = [accX, accXstar, -obj]
% b = [accX, accXstar, -obj] (mejor = mayor valor en este vector)
    for k = 1:numel(a)
        if a(k) > b(k), tf = true;  return; end
        if a(k) < b(k), tf = false; return; end
    end
    tf = false; % iguales -> no mejor
end
