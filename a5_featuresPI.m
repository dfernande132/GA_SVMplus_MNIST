%% Script para obtener PI para las muestras
% Primero se binariza las imagenes en base a un umbral 
% Segundo se cuentan los pixels blancos rodeados de negro (agujeros)
% Tercero se cuentan los pixels blancos
% Esas dos caracteristicas será la PI
clear;clc;
%% Script para convertir a binario los datos de train200.mat

% 1. Cargar los datos de entrenamiento
fichero='train200.mat';
load(fichero);  % Debe cargar la estructura 'train'

% 2. Definir el umbral para la binarización
umbral1 = 0.2;  % Ajusta de umbral
umbral2 = 0.25; % Ajusta de umbral
umbral3 = 0.3;  % Ajusta de umbral
umbral4 = 0.35; % Ajusta de umbral
umbral0  = 0.8;  % Ajusta umbral tiza

% 3. Convertir X_train a una matriz binaria (0 y 1)
% Se aplica el umbral a cada elemento de la matriz.
binary_X_train0 = double(train.X_train > umbral0);
binary_X_train1 = double(train.X_train > umbral1);
binary_X_train2 = double(train.X_train > umbral2);
binary_X_train3 = double(train.X_train > umbral3);
binary_X_train4 = double(train.X_train > umbral4);

% 4. Actualizar la estructura train con la nueva X_train binaria
% Solo descomentar si se quiere guardar la matriz binaria
train.X_train = binary_X_train4;
% La matriz y_train se mantiene sin cambios.
% 5. Guardar la estructura en el archivo train_binary.mat
save('train_binary.mat', 'train');

fprintf('Conversión a binario completada. Los datos se han guardado en train_binary.mat.\n');

%% Contamos los blancos

n = size(binary_X_train1,1);
% Prealocar la matriz de resultados: cada fila [#1s, #agujeros]
result = zeros(n, 5);

for i = 1:n
    % Reconfigurar la fila en una imagen 10x10
    img  = reshape(train.X_train(i, :), [10, 10]);
    img0 = reshape(binary_X_train0(i, :), [10, 10]);
    img1 = reshape(binary_X_train1(i, :), [10, 10]);
    img2 = reshape(binary_X_train2(i, :), [10, 10]);
    img3 = reshape(binary_X_train3(i, :), [10, 10]);
    img4 = reshape(binary_X_train4(i, :), [10, 10]);
    
    % Contar el número de 1 (píxeles blancos)
    ones_count = sum(img0(:));
    
    % Encontrar componentes conexas de píxeles 0 usando conectividad 8.
    % La función bwconncomp opera sobre imágenes lógicas, así que ~img convierte 1 en 0 y viceversa,
    % de modo que obtenemos componentes de píxeles negros (0 en la imagen original).
    CC1 = bwconncomp(~img1, 8);
    CC2 = bwconncomp(~img2, 8);
    CC3 = bwconncomp(~img3, 8);
    CC4 = bwconncomp(~img4, 8);

    holes_count1 = 0;
    holes_count2 = 0;
    holes_count3 = 0;
    holes_count4 = 0;
    
    % Evaluar cada componente:
    for j = 1:CC1.NumObjects
        comp = CC1.PixelIdxList{j};
        % Solo se cuentan componentes de tamaño 1, 2 o 3.
        if numel(comp) >= 1 && numel(comp) <= 3
            % Convertir índices lineales a coordenadas (fila, columna)
            [rows, cols] = ind2sub(size(img1), comp);
            % Verificar que ninguno de los píxeles toca el borde (fila o columna 1 o 10)
            if all(rows > 1 & rows < 10 & cols > 1 & cols < 10)
                holes_count1 = holes_count1 + 1;
            end
        end
    end

    % Evaluar cada componente:
    for j = 1:CC2.NumObjects
        comp = CC2.PixelIdxList{j};
        % Solo se cuentan componentes de tamaño 1, 2 o 3.
        if numel(comp) >= 1 && numel(comp) <= 3
            % Convertir índices lineales a coordenadas (fila, columna)
            [rows, cols] = ind2sub(size(img2), comp);
            % Verificar que ninguno de los píxeles toca el borde (fila o columna 1 o 10)
            if all(rows > 1 & rows < 10 & cols > 1 & cols < 10)
                holes_count2 = holes_count2 + 1;
            end
        end
    end

    % Evaluar cada componente:
    for j = 1:CC3.NumObjects
        comp = CC3.PixelIdxList{j};
        % Solo se cuentan componentes de tamaño 1, 2 o 3.
        if numel(comp) >= 1 && numel(comp) <= 3
            % Convertir índices lineales a coordenadas (fila, columna)
            [rows, cols] = ind2sub(size(img3), comp);
            % Verificar que ninguno de los píxeles toca el borde (fila o columna 1 o 10)
            if all(rows > 1 & rows < 10 & cols > 1 & cols < 10)
                holes_count3 = holes_count3 + 1;
            end
        end
    end


    % Evaluar cada componente:
    for j = 1:CC4.NumObjects
        comp = CC4.PixelIdxList{j};
        % Solo se cuentan componentes de tamaño 1, 2 o 3.
        if numel(comp) >= 1 && numel(comp) <= 3
            % Convertir índices lineales a coordenadas (fila, columna)
            [rows, cols] = ind2sub(size(img4), comp);
            % Verificar que ninguno de los píxeles toca el borde (fila o columna 1 o 10)
            if all(rows > 1 & rows < 10 & cols > 1 & cols < 10)
                holes_count4 = holes_count4 + 1;
            end
        end
    end    

    
    % Guardar los resultados para la imagen actual:
    result(i, :) = [ones_count, holes_count1,holes_count2,holes_count3,holes_count4];
       
    % % Si se encontró al menos un agujero, mostrar la imagen
    % if holes_count > 0
    %     figure;
    %     imshow(img, []);  % '[]' para ajustar el rango de visualización
    %     title(sprintf('Imagen %d: agujeros = %d', i, holes_count));
    % end
end

% Modificamos las etiquetas, 5-> -1 y 8 -> 1
train.y_train(train.y_train == 5) = -1;
train.y_train(train.y_train == 8) = 1;
% Guardamos datos de entrenamiento PI
train.PI_train=result;
% Guardar la estructura en el archivo
% save(fichero, 'train');
% fprintf('Obtenidos los datos para PI. Los datos se han guardado.\n');
