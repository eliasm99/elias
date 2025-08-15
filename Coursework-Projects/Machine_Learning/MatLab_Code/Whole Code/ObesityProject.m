%% Data pre-processing and understanding
% Loading and exploring the dataset
obesity=readtable('ObesityDataSet_raw_and_data_sinthetic.csv');

% Display the first 5 rows of the dataset and dimensions (rows,columns)
disp('Overview');
disp(obesity(1:5, :)); 
disp('Shape')
disp(size(obesity)); 

% Dividing line between sections
disp(repmat('-', 1, 200));

% Display summary of the data
disp('Summary Obesity Statistics:');
summary(obesity);

% Create a target column BMI that is continuous and suitable for
% regression model
% BMI is calculated by dividing weight with square of height

bmi_calc = obesity.Weight ./ (obesity.Height .^ 2);
obesity.bmi = bmi_calc; % Add the column to the table

% Divide into features and target (Obesity category and bmi is the target)
features=obesity(:,1:end-2);
target=obesity(:,end-1:end);

% From 16 features we have 8 categorical and 8 numerical columns, so we
% convert the non-numerical columns to numeric for uniformity in the feature space 

% Get string and categorical format columns
nonnumericcols = varfun(@iscellstr,features,'OutputFormat','uniform') | ...
                     varfun(@iscategorical,features,'OutputFormat','uniform');

% Get the names of those columns
nonnumericcolnames = features.Properties.VariableNames(nonnumericcols);

% Convert non-numerical columns to numeric
for i = 1:length(nonnumericcolnames)
    colname = nonnumericcolnames{i};
    % Convert all columns to categorical
    if ~iscategorical(features.(colname))
        features.(colname) = categorical(features.(colname));
    end
    % Convert to numeric
    features.(colname) = double(features.(colname));
end

% Normalize the feature values between 0 and 1 (the normalize command uses
% the max and min values for rescaling the data)
normfeatures = normalize(features, 'range');

% Show the unique classifications of the target column
categoryNamesran = unique(target.NObeyesdad);

% Reordering categories for better visualization
categoryNames = categoryNamesran([1,2,6,7,3,4,5]);

% Transform the target column to categorical as well
target.NObeyesdad = categorical(target.NObeyesdad);

cats=categoryNames;
counts = countcats(target.NObeyesdad);

% Plotting the distribution of target values (NObeyesdad)
figure;
bar(counts, 'FaceColor', [0.2 0.6 0.8]);
title('Distribution of Target Variable', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'xticklabel', cats, 'FontSize', 12);
xlabel('Obesity Categories', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
grid on;

% The dataset is balanced across all the target categories as expected,
% since the dataset is synthetic (SMOTE technique applied)

% Explore the bmi ranges for the corresponding categories.

% Get unique categories in NObeyesdad
categories = unique(target.NObeyesdad); % Unique category names

% Initialize a table to store BMI ranges
bmiRanges = array2table(zeros(length(categories), 2), ...
    'VariableNames', {'Min_BMI', 'Max_BMI'}, ...
    'RowNames', categoryNames);

% Loop through each category and calculate the BMI range
for i = 1:length(categories)
    % Filter BMI values for the current category
    categoryBMIs = target.bmi(target.NObeyesdad == categoryNames{i});
    
    % Calculate minimum and maximum BMI
    bmiRanges.Min_BMI(i) = min(categoryBMIs);
    bmiRanges.Max_BMI(i) = max(categoryBMIs);
end

% Display the BMI ranges
disp('BMI Ranges for Each Category:');
disp(bmiRanges);

% Create a figure for the histogram
figure;
tiledlayout('flow'); % Create a tiled layout for multiple subplots
% Assign specific colors for each category
colors = lines(length(categories)); % Use distinct colors from a colormap

% Plot histogram for each category
for i = 1:length(categories)
    % Filter BMI values for the current category
    categoryBMIs = target.bmi(target.NObeyesdad == categoryNames{i});
    
    % Create a histogram subplot
    nexttile;
    histogram(categoryBMIs, 'FaceColor', colors(i,:)); % Random color for each category
    title(categoryNames{i}, 'Interpreter', 'none'); % Title for each subplot
    xlabel('BMI');
    ylabel('Frequency');
end
sgtitle('Histograms of BMI by NObeyesdad Categories');

% The categories bmi values mostly follow a normal distribution, showing no
% skewness.

% We notice that the bmi ranges for each category overlap, meaning we
% cannot depend fully on bmi values as target value for training our model.

% Since we can not use bmi values as a target value without heavy
% preprocessing we are ditching the initial plan of using logistic
% regression, since we have a categorical and not a continuous target. So
% the 2 ML models that are used for this project will be Random Forest and
% Decision Tree.

%% ML methods to use-compare: Random Forests and Decision Tree
%% Method 1: Random Forests

% Setting a random seed to reproduce the same results  
rng(1);

% Split the data into training (80%) and testing (20%) sets
cv = cvpartition(target.NObeyesdad, 'HoldOut', 0.2); % Stratified split
trainIdx = training(cv);  % Logical index for training data
testIdx = test(cv);       % Logical index for testing data

% Stratified split was used to preserve the distribution of the target
% variable

% Training and testing subsets
X_train = normfeatures(trainIdx, :);
y_train = target.NObeyesdad(trainIdx);
X_test = normfeatures(testIdx, :);
y_test = target.NObeyesdad(testIdx);

% Hyperparameter grid for Random Forest
nTrees = round(linspace(250, 350, 5));
minLeafSizes = round(linspace(1, 3, 3));
maxSplits = round(linspace(350, 450, 5));

% These ranges were defined after some intermediate results on wider ranges

% Preallocate results array
numCombinations = numel(nTrees) * numel(minLeafSizes) * numel(maxSplits);
results(numCombinations) = struct('NumTrees', [], ...
                                   'MinLeafSize', [], ...
                                   'MaxSplits', [], ...
                                   'CVError', []);
% Preallocating a results array makes the code run much faster since it
% reduces the time needed to change the array dimensions on each iteration

resultIdx = 1; % Index for results array

% Initialize variables for tracking the best model
lowestError = Inf; % High value to be replaced with the first iteration 
bestParams = struct();

% Perform grid search with k-fold cross-validation
k = 5; % Number of folds
% Chose 5 folds instead of standard 10 because of the number of the models
% we had to train (5*3*5)*5 folds = 375 models that is already a high
% number of models.
disp('Starting grid search...');
for nTree = nTrees
    for minLeaf = minLeafSizes
        for maxSplit = maxSplits
            % Cross-validation for Random Forest
            cvError = 0; % Initialize cumulative cross-validation error
            cvInner = cvpartition(y_train, 'KFold', k); % Create KFold partition

            for fold = 1:k
                % Training and validation subsets for this fold
                trainIdxInner = training(cvInner, fold);
                testIdxInner = test(cvInner, fold);

                X_trainInner = X_train(trainIdxInner, :);
                y_trainInner = y_train(trainIdxInner);
                X_testInner = X_train(testIdxInner, :);
                y_testInner = y_train(testIdxInner);

                % Train Random Forest for this fold
                model = TreeBagger(nTree, X_trainInner, y_trainInner, ...
                                   'OOBPrediction', 'off', ...
                                   'Method', 'classification', ...
                                   'MinLeafSize', minLeaf, ...
                                   'MaxNumSplits', maxSplit);

                % Predict on validation set
                y_predInner = predict(model, X_testInner);
                y_predInner = categorical(y_predInner); % Convert to categorical

                % Calculate validation error for this fold
                foldError = sum(y_predInner ~= y_testInner) / length(y_testInner);
                cvError = cvError + foldError; % Accumulate error
            end

            % Average cross-validation error over k folds
            cvError = cvError / k;

            % Store the results in the preallocated array
            results(resultIdx).NumTrees = nTree;
            results(resultIdx).MinLeafSize = minLeaf;
            results(resultIdx).MaxSplits = maxSplit;
            results(resultIdx).CVError = cvError;
            resultIdx = resultIdx + 1;

            % Update the best model parameters if current configuration is better
            if cvError < lowestError
                lowestError = cvError;
                bestParams = struct('NumTrees', nTree, ...
                                    'MinLeafSize', minLeaf, ...
                                    'MaxSplits', maxSplit);
            end
        end
    end
end

disp('Grid search complete.');
disp('Best Hyperparameters:');
disp(bestParams);

%% Method 2: Decision Trees

% Hyperparameter grid for Decision Tree
minLeafSizes = round(linspace(1, 5, 5));
maxSplits = round(linspace(50, 150, 5));


% Preallocate results array for Decision Tree
dtResults(numel(minLeafSizes) * numel(maxSplits)) = struct('MinLeafSize', [], ...
                                                           'MaxSplits', [], ...
                                                           'CVError', []);
dtResultIdx = 1; % Index for Decision Tree results

% Initialize variables for tracking the best model
lowestErrorDT = Inf;
bestDTParams = struct();

% Perform grid search for Decision Tree with k-fold cross-validation
k = 5; % Number of folds

disp('Starting grid search for Decision Tree...');
for minLeaf = minLeafSizes
    for maxSplit = maxSplits
        % Cross-validation for Decision Tree
        cvErrorDT = 0; % Initialize cumulative cross-validation error
        cvInner = cvpartition(y_train, 'KFold', k); % Create KFold partition

        for fold = 1:k
            % Training and validation subsets for this fold
            trainIdxInner = training(cvInner, fold);
            testIdxInner = test(cvInner, fold);

            X_trainInner = X_train(trainIdxInner, :);
            y_trainInner = y_train(trainIdxInner);
            X_testInner = X_train(testIdxInner, :);
            y_testInner = y_train(testIdxInner);

            % Train Decision Tree for this fold
            dtModel = fitctree(X_trainInner, y_trainInner, 'MinLeafSize', minLeaf, ...
                               'MaxNumSplits', maxSplit);

            % Predict on validation set
            y_predInner = predict(dtModel, X_testInner);
            y_predInner = categorical(y_predInner); % Convert to categorical

            % Calculate validation error for this fold
            foldErrorDT = sum(y_predInner ~= y_testInner) / length(y_testInner);
            cvErrorDT = cvErrorDT + foldErrorDT; % Accumulate error
        end

        % Average cross-validation error over k folds
        cvErrorDT = cvErrorDT / k;

        % Store the results for Decision Tree in the preallocated array
        dtResults(dtResultIdx).MinLeafSize = minLeaf;
        dtResults(dtResultIdx).MaxSplits = maxSplit;
        dtResults(dtResultIdx).CVError = cvErrorDT;
        dtResultIdx = dtResultIdx + 1;

        % Update the best Decision Tree hyperparameters if current configuration is better
        if cvErrorDT < lowestErrorDT
            lowestErrorDT = cvErrorDT;
            bestDTParams = struct('MinLeafSize', minLeaf, ...
                                  'MaxSplits', maxSplit);
        end
    end
end

disp('Grid search for Decision Tree complete.');
disp('Best Hyperparameters for Decision Tree:');
disp(bestDTParams);

%% Exporting the results of grid search to train and evaluate the best models

% Save datasets and hyperparameters
save('final_model_data.mat', 'X_train', 'y_train', 'X_test', 'y_test', ...
     'bestParams', 'bestDTParams');
%% Final Models training and evaluation

% Timing and training Random Forest final model
% We use the hyperparameters with the smallest CV error, prioritizing
% generalization over accuracy for the selected model

tic; % Initializing the timer
finalModel = TreeBagger(bestParams.NumTrees, X_train, y_train, ...
                        'OOBPrediction', 'off', ...
                        'Method', 'classification', ...
                        'MinLeafSize', bestParams.MinLeafSize, ...
                        'MaxNumSplits', bestParams.MaxSplits);
trainingTimeRF = toc; % Collecting the training time in seconds

% Timing and training Decision Tree final model

tic; % Initializing the timer
finalDTModel = fitctree(X_train, y_train, ...
                        'MinLeafSize', bestDTParams.MinLeafSize, ...
                        'MaxNumSplits', bestDTParams.MaxSplits);
trainingTimeDT = toc; % Collecting the training time in seconds

% Accuracy of models

% Predictions for Random Forest
y_predRF = predict(finalModel, X_test);
y_predRF = categorical(y_predRF); % Convert predictions to categorical

% Predictions for Decision Tree
y_predDT = predict(finalDTModel, X_test);
y_predDT = categorical(y_predDT); % Convert predictions to categorical

% Confusion Matrices

% Confusion matrix for Random Forest
confusionMatrixRF = confusionmat(y_test, y_predRF);
figure;
confusionchart(y_test, y_predRF, 'Title', 'Random Forest Confusion Matrix');

% Confusion matrix for Decision Tree
confusionMatrixDT = confusionmat(y_test, y_predDT);
figure;
confusionchart(y_test, y_predDT, 'Title', 'Decision Tree Confusion Matrix');

% Different evaluation metrics 

% Compute Precision, Recall, and F1 for Random Forest
[~, ~, ~] = multiclassMetrics(confusionMatrixRF);

% Compute Precision, Recall, and F1 for Decision Tree
[~, ~, ~] = multiclassMetrics(confusionMatrixDT);

% Function for precision, recall, and F1
function [precision, recall, f1] = multiclassMetrics(confMat)
    precision = diag(confMat) ./ sum(confMat, 1)'; % True Positives / Predicted Positives
    recall = diag(confMat) ./ sum(confMat, 2);    % True Positives / Actual Positives
    f1 = 2 * (precision .* recall) ./ (precision + recall); % Harmonic Mean
    precision(isnan(precision)) = 0; % Handle division by zero
    recall(isnan(recall)) = 0;
    f1(isnan(f1)) = 0;
end

% Overall metrics for Random Forest
overallAccuracyRF = sum(diag(confusionMatrixRF)) / sum(confusionMatrixRF(:)) * 100; % Accuracy
[precisionRF, recallRF, f1ScoreRF] = multiclassMetrics(confusionMatrixRF);
overallPrecisionRF = mean(precisionRF, 'omitnan'); % Mean precision
overallRecallRF = mean(recallRF, 'omitnan');       % Mean recall
overallF1ScoreRF = mean(f1ScoreRF, 'omitnan');     % Mean F1 score

% Display metrics for Random Forest
disp('Random Forest Metrics:');
disp(['Random Forest Training Time: ', num2str(trainingTimeRF), ' seconds']);
disp(['Accuracy: ', num2str(overallAccuracyRF), '%']);
disp(['Precision: ', num2str(overallPrecisionRF)]);
disp(['Recall: ', num2str(overallRecallRF)]);
disp(['F1 Score: ', num2str(overallF1ScoreRF)]);

% Overall metrics for Decision Tree
overallAccuracyDT = sum(diag(confusionMatrixDT)) / sum(confusionMatrixDT(:)) * 100; % Accuracy
[precisionDT, recallDT, f1ScoreDT] = multiclassMetrics(confusionMatrixDT);
overallPrecisionDT = mean(precisionDT, 'omitnan'); % Mean precision
overallRecallDT = mean(recallDT, 'omitnan');       % Mean recall
overallF1ScoreDT = mean(f1ScoreDT, 'omitnan');     % Mean F1 score

% Display metrics for Decision Tree
disp('Decision Tree Metrics:');
disp(['Decision Tree Training Time: ', num2str(trainingTimeDT), ' seconds']);
disp(['Accuracy: ', num2str(overallAccuracyDT), '%']);
disp(['Precision: ', num2str(overallPrecisionDT)]);
disp(['Recall: ', num2str(overallRecallDT)]);
disp(['F1 Score: ', num2str(overallF1ScoreDT)]);

% ROC Curves

% Random Forest ROC
figure;
hold on;
classes = unique(y_test);
for i = 1:numel(classes)
    % Create binary labels for one-vs-all
    y_trueBinary = (y_test == classes(i));
    [~, scoresRF] = predict(finalModel, X_test); % Probabilities from RF
    [X, Y, ~, AUC] = perfcurve(y_trueBinary, scoresRF(:, i), true);
    plot(X, Y, 'DisplayName', ['Class ', char(classes(i)), ' AUC: ', num2str(AUC)]);
end
title('Random Forest ROC Curves');
legend;
hold off;

% Decision Tree ROC
figure;
hold on;
for i = 1:numel(classes)
    % Create binary labels for one-vs-all
    y_trueBinary = (y_test == classes(i));
    [~, scoresDT] = predict(finalDTModel, X_test); % Probabilities from Decision Tree
    [X, Y, ~, AUC] = perfcurve(y_trueBinary, scoresDT(:, i), true);
    plot(X, Y, 'DisplayName', ['Class ', char(classes(i)), ' AUC: ', num2str(AUC)]);
end
title('Decision Tree ROC Curves');
legend;
hold off;

%% Export data for different visualization tools

% All of the graphs are produced from the data in this matlab code in a
% python environment using more advanced visualization libraries

normfeatures = splitvars(normfeatures);
% Save normalized features
writetable(normfeatures, 'normfeatures.csv');

% Save target data
writetable(target, 'target.csv');

% Save confusion matrices and metrics
save('model_metrics.mat', 'confusionMatrixRF', 'confusionMatrixDT', 'precisionRF', 'recallRF', 'f1ScoreRF', ...
     'precisionDT', 'recallDT', 'f1ScoreDT');

% Export data for plotting the ROC curves
csvwrite('rf_probabilities.csv', scoresRF);
csvwrite('dt_probabilities.csv', scoresDT);

y_test_table = cell2table(cellstr(y_test));  % Convert categorical to cell array of strings
writetable(y_test_table, 'true_labels.csv');  % Save as CSV

% We ignore the csvwrite command warnings as this format is preffered for
% data processing and usage in a python environment