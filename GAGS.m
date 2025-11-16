%% Optimizer SMV+ parameters FUNCTION
% Based on genetic algorithms
% 
% GOP_LUPI (Grid Optimizer Parallel)
% Function created by Jose Daniel Fernandez - October 2024
% Function implements a genetic algorithm (GA) to obtain the
% best parameters for the SMV+ LUPI model with m x n signals

% function GOP_LUPI 
function [bestResult, llam] = GAGS(fv,fvStar,lbl,depth,B)
clc;
% if nargin == 3
%     depth=1;
% elseif nargin ~= 4
%     error('Incorrect number of parameters. The function must receive 3 or 4 parameters: fv,fvStar,lbl,<depth>');
% end

tic;
ruta="out/";
currentDateTime = datestr(now, 'yyyymmdd_HHMM'); % for the output files

% Define GA parameters
% Define limits of the variable for GridSearch
% GA GRID  PARAMETERS
  lb = [1e-3, 1e-4, 1e-2, 1e-2];
    ub = [1e+3, 1e+2, 1e+2, 1e+2];

CparamPlus_ini  =  [1e-3, 1e+3];
gammaParam_ini  =  [1e-1, 1e+2];
sgmPlus_ini    =   [1e-2, 1e+2];  
sgmStar_ini    =   [1e-2, 1e+2];  

switch depth
    case 1
        numpartesC=4;
        numpartesG=4;
        numpartesSP=2;
        numpartesSS=2;
    case 2
        CparamPlus_ini  =  [.01, 100];
        gammaParam_ini  =  [.001, 10];
        sgmPlus_ini    =   [.01, 10];  % Small sigmas-->overfitting
        sgmStar_ini    =   [.01, 10];  % Large Sigmas-->smoothing
        numpartesC=4;
        numpartesG=4;
        numpartesSP=3;
        numpartesSS=3;
    case 3
        CparamPlus_ini  =  [.001, 100];
        gammaParam_ini  =  [.0001, 10];
        sgmPlus_ini    =   [.01, 10];  % Small sigmas-->overfitting
        sgmStar_ini    =   [.01, 10];  % Large Sigmas-->smoothing
        numpartesC=5;
        numpartesG=5;
        numpartesSP=3;
        numpartesSS=3;
end
 

numloop=numpartesC*numpartesG*numpartesSP*numpartesSS;
CparamPlus_pt = logspace(log10(CparamPlus_ini(1)), log10(CparamPlus_ini(2)), numpartesC+ 1);
gammaParam_pt = logspace(log10(gammaParam_ini(1)), log10(gammaParam_ini(2)), numpartesG + 1);
sgmPlus_pt = logspace(log10(sgmPlus_ini(1)), log10(sgmPlus_ini(2)), numpartesSP + 1);
sgmStar_pt = logspace(log10(sgmStar_ini(1)), log10(sgmStar_ini(2)), numpartesSS + 1);
rango=[CparamPlus_pt';gammaParam_pt';sgmPlus_pt';sgmStar_pt'];

% GA PARAMETERS
pop_size = 40*depth; % population size
generations =B/20; % number of generations 
parent_num=20*depth ; % number of parents/generation. x parents means x children. An even number must be selected.
adopted=1 ; % Number of adopted children
num_vars = 4; % number of decision variables
crossover_prob = .9; % probability of crossing
mutation_prob = 0.02; % probability of mutation
cross_value=0.2; % Value to multiply the total arithmetic crossing
similarity=.9; % The similarity between parents and children
message=1000; % displays a message on the screen every X generations
gen_conv=25; % Number of generations to check convergence
fitness_order=[-6 -7 5]; % column 8 is random =[-6 -7 8]
coldec = [3, 4, 4, 4]; % Number of decimals for parameters population
decimales=[3 3 2 2]; % Round convergence control
llam=generations*parent_num;
% Expected time 
factor_time=25;
exp_time=numloop*(generations*factor_time)*(parent_num/10);
exp_time=exp_time/10; % pararell optimization

% files
ficheroGOP=ruta+'out_GOP_'+string(currentDateTime)+'.mat';
ficheroTXT=ruta+'out_TXT_'+string(currentDateTime)+'.txt';
ficheroPOP=ruta+'out_POP_'+string(currentDateTime)+'.mat';
fileID = fopen(ficheroTXT, 'w');
pop_inicial=[1,1,1,1,0,0,0];


% The population matrix has this structure
% column 1= decision variable 'CparamPlus'
% column 2= decision variable 'gammaParam'
% column 3= decision variable 'sgmPlus'
% column 4= decision variable 'sgmStar' 
% column 5= objective function 'valPlus' 
% column 6= numSucc/tot success in the decision variable
% column 7= numSuccCorr/tot success in the correction variable


str = sprintf('*********** Running GA GridOptimizerParallel**************\n');
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(2,str);          % Escribe en la consola
str = sprintf('Expected time in sec: %d s. with parallel workers\n', round(exp_time));
    fprintf(str);          % Escribe en la consola   
str = sprintf(string(generations)+ ' generations of '+string(pop_size)+ ' populations and total loop: %d\n',numloop);
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola
str = sprintf('The lbl matrix has %d rows: %d positives vectors and %d negatives.\n', length(lbl), sum(lbl == 1), sum(lbl == -1));
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola

% create pop_loop & pop_inicial variables
pop_loop=[];
% pop_inicial=[];

% Generate all combinations of indices for params
[A, B, C, D] = ndgrid(1:numpartesC, 1:numpartesG, 1:numpartesSP, 1:numpartesSS);
% Convert the matrices into column vectors
A = A(:);
B = B(:);
C = C(:);
D = D(:);

parfor bucle=1:numloop % ****** Paralleling computing *******

    % Recovers the ranges according to the loop number
    idxC=A(bucle);
    idxG=B(bucle);
    idxSP=C(bucle);
    idxSS=D(bucle);
    CparamPlus=[CparamPlus_pt(idxC),CparamPlus_pt(idxC+1)];
    gammaParam=[gammaParam_pt(idxG),gammaParam_pt(idxG+1)];
    sgmPlus=[sgmPlus_pt(idxSP),sgmPlus_pt(idxSP+1)];
    sgmStar=[sgmStar_pt(idxSS),sgmStar_pt(idxSS+1)];

% *******  Population initialization  ******* 
limits = [CparamPlus' gammaParam' sgmPlus' sgmStar'];
% str = sprintf('Begin new loop: %d/%d ........ \n',loops,totalloop);
str = sprintf('Begin new loop: %d/%d ........ \n',bucle,numloop);
    %fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola
% The population is created randomly on a logarithmic scale with the rndlog function
pop = zeros(pop_size, num_vars);
for i = 1:4
    for j = 1:pop_size
        pop(j, i) = rndlog(limits(1, i), limits(2, i));
    end
end

% Rounds decimals
pop=redondeoDec(pop,coldec);

% Eval objective function
fitness = evalua_fitness(pop,fv,fvStar,lbl);
pop=[pop fitness]; 

% *******  Main loop of the genetic algorithm ******* 
for gen = 1:generations
    if mod(gen, message)==0
        str = sprintf('Gen %d/%d. C %0.3f, Gamma %0.3f, SigmaP %0.3f, SigmaS %0.3f, Dec %0.3f, Corr %0.3f \n',...
            gen, generations,pop(1,1),pop(1,2),pop(1,3),pop(1,4),pop(1,6),pop(1,7));
            %fprintf(fileID, str);  % Escribe en el archivo
            fprintf(str);          % Escribe en la consola
    end

    % Parent selection through binary tournament
    parent_selection = torneo_seleccion(pop, parent_num);
    
    % Total arithmetic crossing of the selected parents
    parents=pop(parent_selection,1:num_vars);
    children = cruce(parents, crossover_prob, cross_value, similarity); 
    
    % Mutation of children generated
    children = mutacion(children, mutation_prob,limits); % mutacion cambiando numero aleatoriamente
    
    % Adopted children
    c_adopted=genera_hijos_adoptivos(adopted,num_vars,limits);    
    children=[children;c_adopted];

    % Eval objective function of generated children
    children=redondeoDec(children,coldec);
    fitness = evalua_fitness(children,fv,fvStar,lbl);
    children=[children fitness]; % Add values of the objective functions
    
    % Replacement of the worst individuals in the population with the children generated
    pop = replazamiento_elitista(pop, children, pop_size, limits, coldec, num_vars, fitness_order, fv, fvStar, lbl);

    % Convergence Control
    if (mod(gen, gen_conv)==0 && (generations-gen)>20)
        pop = convergencia(pop, pop_size, limits, num_vars,fitness_order,decimales,fv,fvStar,lbl,coldec);
    end
end
    % Save the best individual in pop_loop
pop_loop=[pop_loop;pop(1,:)];
filas_condicion = pop(:, 6) == 1 & pop(:, 7) == 1;
add_pop=pop(filas_condicion,:);
if ~isempty(add_pop)
pop_inicial=[pop_inicial;add_pop];
end
str = sprintf('Loop %d completed after %d gen. Success sigma %2.1f and sigma* %2.1f percent \n',bucle, gen,(round(pop(1,6)*1000)/10), (round(pop(1,7)*1000)/10));
    %fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola
% currentDateTime = datestr(now, 'yyyymmdd_HHMMSS'); % Formato: 'añoMesDía_HoraMinutoSegundo'

end % END PARFOR

% Write files and return solution
time_exec = toc;

str = sprintf('Added '+string(size(pop_inicial, 1)-1)+' models with successful classification in both spaces\n');
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola
str =sprintf('The execution time was %d seconds\n', round(time_exec));
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola
str = sprintf('***** GA Finished. Save grids result in '+ficheroGOP+' ***** \n');
    fprintf(fileID, str);  % Escribe en el archivo
    fprintf(str);          % Escribe en la consola

save(ficheroGOP,'pop_loop','rango');
save(ficheroPOP,'pop_inicial');
fclose(fileID);

out_GOP=pop_loop; 
% out_POP=pop_inicial;
out_GOP=sortrows(out_GOP,fitness_order); 
bestResult=out_GOP(1,:);

end % ******** End of main GA function *******




%% GA Functions 
% ***********************************************************
% ***********************************************************
%                     GA auxiliar functions
% ***********************************************************
% ***********************************************************
%
% Set of auxiliary functions for the genetic algorithm, including 
% evaluation, mutation, crossover, and selection functions, among others.

% Eval objective function
function [fitness] = evalua_fitness(pop,fv,fvStar,lbl)
    n=size(pop, 1);
    fitness=zeros(n,3);   
    for i=1 : n
        % Call fitness function 
        [valPlus, ~, ~, ~, result] = QFOM_LUPI(fv, fvStar, lbl, pop(i,1),pop(i,2),pop(i,3),pop(i,4));
        fitness(i,:)=[valPlus,result(4:5)];
    end
    
end


% Parent selection through binary tournament
function [selec_parent] = torneo_seleccion(pop, parent_num)
    pop_size=numel(pop(:,1));
    selec_parent = zeros(parent_num, 1);
    for i = 1:parent_num
        selected = randperm(pop_size, 2);
        if pop(selected(1), 7) > pop(selected(2), 7)
            selec_parent(i) = selected(1);
        else
            selec_parent(i) = selected(2);
        end
    end
end


% Total arithmetic crossover of the selected parents
function [children] = cruce(parents, crossover_prob, cross_value, similarity)
    num_parents = size(parents, 1);
    children = zeros(size(parents));
    for i = 1:2:num_parents % Parents' choice in pairs
        if rand < crossover_prob
            % Total Arithmetic Crossover
            % The crossover is carried out in the logarithmic space
            logPadre1 = log(parents(i,:));
            logPadre2 = log(parents(i+1,:));    
            combinaPadre1 = cross_value .* logPadre1 + (1-cross_value) .* logPadre2;
            children(i,:) = exp(combinaPadre1);
            combinaPadre2 = cross_value .* logPadre2 + (1-cross_value) .* logPadre1;
            children(i+1,:) = exp(combinaPadre2);
        elseif similarity ~=1
            % Copy of uncrossed parents with small variation
            vbajo=similarity;
            valto=1+(1-similarity);
            % Generate similarity matrix
            percent = vbajo + (valto - vbajo) * rand(2,4);
            children(i,:) = parents(i,:) .* percent(1,:);
            children(i+1,:) = parents(i+1,:) .* percent(2,:);
        end
    end
    children(children(:,1)==0,:)=[];
end


% Mutation Function with the Specified Probability
function [mutated] = mutacion(children, mutation_prob, limits)
    [m, n] = size(children);
    mutated = children;
    for i = 1:m
        for j = 1:n
            if rand < mutation_prob
                mutated(i,j) = limits(1,j) + rand*(limits(2,j)-limits(1,j));
            end
        end
    end
end


% Adoptive Offspring Generation Function
function c_adopted=genera_hijos_adoptivos(adopted,num_vars,limits)
    c_adopted = zeros(adopted, num_vars);
    for i = 1:num_vars
        for j = 1:adopted
            c_adopted(j, i) = rndlog(limits(1, i), limits(2, i));
        end
    end   
end


% Elitist Replacement Function
function [pop] = replazamiento_elitista(pop, children, pop_size, limits, coldec, num_vars,fitness_order,fv,fvStar,lbl)
    pop=[children;pop];
    pop=redondeoDec(pop, coldec); % Round population Matrix
    % Twins are eliminated from the population
    pop = unique(pop, 'rows');
    % Complete new population if duplicates have been removed
    faltan=pop_size-size(pop,1);
    if faltan>0
        % Population is created randomly on a logarithmic scale using the rndlog function
        news = zeros(faltan, num_vars);
        for i = 1:4
            for j = 1:faltan
                news(j, i) = rndlog(limits(1, i), limits(2, i));
            end
        end        
        news=redondeoDec(news, coldec); % Round news elements Matrix
        fitness = evalua_fitness(news,fv,fvStar,lbl); % Fitness function
        news=[news fitness]; 
        pop=[unicos;news];
        pop = unique(pop, 'rows');
    end
    % Sorted by success and the function value
    % Add random number in population
    numFilas = size(pop, 1);
    pop(:, 8) = rand(numFilas, 1);
    pop=sortrows(pop,fitness_order); 
    pop=pop(1:pop_size,:); % The population is reduced
    pop(:, 8) = [];
end


% Convergence control
function [pop]=convergencia(pop, pop_size, limits, num_vars, fitness_order, decimales, fv, fvStar, lbl,coldec)
    % Initialize the array to store the number of unique elements in each column
    numUnicos = zeros(1, num_vars);
    % Count the unique elements in each of the genes
    for i = 1:num_vars
        columnai = round(pop(:, i),decimales(i));
        numUnicos(i) = numel(unique(columnai));
    end
    % Percentage of unique elements relative to Population total
    desviacion=(pop_size*num_vars -sum(numUnicos))/(pop_size*num_vars); 
    if desviacion>0.9
        % The population that had reached convergence is renewed
        [~, gen_var] = max(numUnicos);
        columna = round(pop(:, gen_var),decimales(gen_var));
        [~, idx] = unique(columna,'stable');
        unicos = pop(idx, :);
        rest=size(unicos,1);
        % Population is created randomly on a logarithmic scale using the rndlog function
        nuevos = zeros(pop_size-rest, num_vars);
        for i = 1:num_vars
            for j = 1:pop_size-rest
                nuevos(j, i) = rndlog(limits(1, i), limits(2, i));
            end
        end        
        nuevos=redondeoDec(nuevos, coldec); % Round Function
        fitness = evalua_fitness(nuevos,fv,fvStar,lbl); % Fitness function
        nuevos=[nuevos fitness]; 
        pop=[unicos;nuevos];
        % str = sprintf('Convergence percent: %2.0f. News %d gens whith param %d \n',...
        %     desviacion*100, pop_size-rest,gen_var);
        %     %fprintf(fileID, str);  % Escribe en el archivo
        %     fprintf(str);          % Escribe en la consola
    else
        % str = sprintf('Convergence percent: %2.0f  \n', desviacion*100);
        %     %fprintf(fileID, str);  % Escribe en el archivo
        %     fprintf(str);          % Escribe en la consola
    end
    % Sorted by success and the function value
    numFilas = size(pop, 1);
    pop(:, 8) = rand(numFilas, 1);
    pop=sortrows(pop,fitness_order);
    pop(:,8) = [];
end


% Function that generates a random number on a logarithmic scale
function num = rndlog(limiteInferior, limiteSuperior)
    % Verify that the limits are positive and greater than zero
    if limiteInferior <= 0 || limiteSuperior <= 0
        error('The limits must be positive and greater than zero.');
    end
    % Convert the limits to a logarithmic scale
    logLimInferior = log10(limiteInferior);
    logLimSuperior = log10(limiteSuperior);
    % Generates a random number between the logarithmic limits
    x = logLimInferior + (logLimSuperior - logLimInferior) * rand();
    % Converts back to the original scale
    num = 10^x;
end


% Function that retains the best populations and saves them as a seed
function save_seed (pop,fitness_order, pop_size, fileID, ficheroPOP)
    load(ficheroPOP);
    tampre=size(pop_inicial, 1);
    if pop_inicial==[1,1,1,1,0,0,0]
        tampre=0;
    end
    pop_inicial=[pop_inicial; pop];
    % Only the population that is 100% successful is recovered
    filas_condicion = pop_inicial(:, 6) == 1 & pop_inicial(:, 7) == 1;
    pop_inicial = pop_inicial(filas_condicion, :);
    tampos=size(pop_inicial, 1);
    if tampos>tampre
        % Use 'unique' to find unique indexes.
        [~, ia, ~] = unique(pop_inicial(:,1:4), 'rows', 'stable');
        pop_inicial = pop_inicial(ia,:);
        % Save in disc
        numFilas = size(pop_size, 1);
        pop_inicial(:, 8) = rand(numFilas, 1);
        pop_inicial=sortrows(pop_inicial,fitness_order);
        pop_inicial(:,8) = [];
        tampos=size(pop_inicial, 1);
        save(ficheroPOP,'pop_inicial');
        str = sprintf('************ Add '+string(tampos-tampre)+' seed *****************\n');
            fprintf(fileID, str);  % Escribe en el archivo
            fprintf(2,str);          % Escribe en la consola
    end
end

% Function that rounds the genes to the specified number of decimals
function round_pop = redondeoDec(pop,coldec) 
    for i = 1:4
        factor = 10^coldec(i);
        pop(:, i) = round(pop(:, i) * factor) / factor;
    end
    round_pop=pop;
end

