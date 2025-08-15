%% Final Models training and evaluation

% First we load the best parameters from grid search
% Load datasets and hyperparameters
load('final_model_data.mat');

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