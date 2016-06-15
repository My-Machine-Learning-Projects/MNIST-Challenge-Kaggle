
%% Artificial Neural Networks for Beginners
% <https://en.wikipedia.org/wiki/Deep_learning Deep Learning> is a very hot
% topic these days especially in computer vision applications and you
% probably see it in the news and get curious. Now the question is, how do
% you get started with it? Today's guest blogger, Toshi Takeuchi, gives us
% a quick tutorial on
% <https://en.wikipedia.org/wiki/Artificial_neural_network artificial
% neural networks> as a starting point for your study of deep learning. 

%% MNIST Dataset
% Many of us tend to learn better with a concrete example. Let me give you
% a quick step-by-step tutorial to get intuition using a popular
% <http://yann.lecun.com/exdb/mnist/index.html MNIST handwritten digit
% dataset>. Kaggle happens to use this very dataset in the
% <https://www.kaggle.com/c/digit-recognizer Digit Recognizer> tutorial
% competition. Let's use it in this example. You can download the
% competition dataset from <https://www.kaggle.com/c/digit-recognizer/data
% "Get the Data"> page:
%
% * train.csv - training data
% * test.csv  - test data for submission
% 
% Load the training and test data into MATLAB, which I assume was
% downloaded into the current folder. The test data is used to generate
% your submissions.

tr = csvread('train.csv', 1, 0);                  % read train.csv
sub = csvread('test.csv', 1, 0);                  % read test.csv

%%
% The first column is the label that shows the correct digit for each
% sample in the dataset, and each row is a sample. In the remaining
% columns, a row represents a 28 x 28 image of a handwritten digit, but all
% pixels are placed in a single row, rather than in the original
% rectangular form. To visualize the digits, we need to reshape the rows
% into 28 x 28 matrices. You can use
% <http://www.mathworks.com/help/matlab/ref/reshape.html reshape> for that,
% except that we need to transpose the data, because |reshape| operates by
% column-wise rather than row-wise.

figure                                          % plot images
colormap(gray)                                  % set to grayscale
for i = 1:25                                    % preview first 25 samples 
    subplot(5,5,i)                              % plot them in 6 x 6 grid
    digit = reshape(tr(i, 2:end), [28,28])';    % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(tr(i, 1)))                    % show the label
end

%% Data Preparation
% You will be using the
% <http://www.mathworks.com/help/nnet/ref/nprtool.html nprtool> pattern
% recognition app from <http://www.mathworks.com/products/neural-network/
% Neural Network Toolbox>. The app expects two sets of data:
%
% * inputs - a numeric matrix, each column representing the samples and
% rows the features. This is the scanned images of handwritten digits.
% * targets - a numeric matrix of 0 and 1 that maps to specific labels that
% images represent. This is also known as a dummy variable. Neural Network
% Toolbox also expects labels stored in columns, rather than in rows.
%
% The labels range from 0 to 9, but we will use '10' to represent '0'
% because MATLAB is indexing is 1-based.
% 
%   1 --> [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]
%   2 --> [0; 1; 0; 0; 0; 0; 0; 0; 0; 0]
%   3 --> [0; 0; 1; 0; 0; 0; 0; 0; 0; 0]
%               :
%   0 --> [0; 0; 0; 0; 0; 0; 0; 0; 0; 1]
% 
% The dataset stores samples in rows rather than in columns, so you need to
% transpose it. Then you will partition the data so that you hold out 1/3
% of the data for model evaluation, and you will only use 2/3 for training
% our artificial neural network model.

n = size(tr, 1);                    % number of samples in the dataset
targets  = tr(:,1);                 % 1st column is |label|
targets(targets == 0) = 10;         % use '10' to present '0'
targetsd = dummyvar(targets);       % convert label into a dummy variable
inputs = tr(:,2:end);               % the rest of columns are predictors

inputs = inputs';                   % transpose input
targets = targets';                 % transpose target
targetsd = targetsd';               % transpose dummy variable

rng(1);                             % for reproducibility 
c = cvpartition(n,'Holdout',n/3);   % hold out 1/3 of the dataset

Xtrain = inputs(:, training(c));    % 2/3 of the input for training
Ytrain = targetsd(:, training(c));  % 2/3 of the target for training
Xtest = inputs(:, test(c));         % 1/3 of the input for testing
Ytest = targets(test(c));           % 1/3 of the target for testing
Ytestd = targetsd(:, test(c));      % 1/3 of the dummy variable for testing

%% Using the Neural Network Toolbox GUI App
% 
% # You can start the Neural Network Start GUI by typing the command
% <http://www.mathworks.com/help/nnet/ref/nnstart.html nnstart>.
% # You then click the Pattern Recognition Tool to open the Neural Network
% Pattern Recognition Tool. You can also use the command
% <http://www.mathworks.com/help/nnet/ref/nprtool.html nprtool> to open it
% directly.
% # Click "Next" in the welcome screen and go to "Select Data". 
% # For |inputs|, select |Xtrain| and for |targets|, select |Ytrain|.
% # Click "Next" and go to "Validation and Test Data". Accept the default
% settings and click "Next" again. This will split the data into 70-15-15
% for the training, validation and testing sets. 
% # In the "Network Architecture", change the value for the number of
% hidden neurons, 100, and click "Next" again. 
% # In the "Train Network", click the "Train" button to start the training.
% When finished, click "Next". Skip "Evaluate Network" and click next.
% # In "Deploy Solution", select "MATLAB Matrix-Only Function" and save the
% generated code. I save it as
% <http://blogs.mathworks.com/images/loren/2015/myNNfun.m myNNfun.m>.
% # If you click "Next" and go to "Save Results", you can also save the
% script as well as the model you just created. I saved the simple script
% as <http://blogs.mathworks.com/images/loren/2015/myNNscript.m
% myNNscript.m>
%
% Here is the diagram of this artificial neural network model you created
% with the Pattern Recognition Tool. It has 784 input neurons, 100 hidden
% layer neurons, and 10 output layer neurons.
% 
% <<network_diagram.png>>
%
% Your model learns through training the weights to produce the correct
% output.
%
% |W| in the diagram stands for _weights_ and |b| for _bias units_, which
% are part of individual neurons. Individual neurons in the hidden layer
% look like this - 784 inputs and corresponding weights, 1 bias unit,
% and 10 activation outputs.
% 
% <<neuron.png>>

%% Visualizing the Learned Weights
% If you look inside |myNNfun.m|, you see variables like |IW1_1| and
% |x1_step1_keep| that represent the weights your artificial neural
% network model learned through training. Because we have 784 inputs and
% 100 neurons, the full layer 1 weights will be a 100 x 784 matrix. Let's
% visualize them. This is what our neurons are learning!



%% Computing the Categorization Accuracy
% Now you are ready to use |myNNfun.m| to predict labels for the heldout
% data in |Xtest| and compare them to the actual labels in |Ytest|. That
% gives you a realistic predictive performance against unseen data. This is
% also the metric Kaggle uses to score submissions.
%
% First, you see the actual output from the network, which shows the
% probability for each possible label. You simply choose the most probable
% label as your prediction and then compare it to the actual label. You
% should see 95% categorization accuracy. 

Ypred = myNNfun(Xtest);             % predicts probability for each label
Ypred(:, 1:5)                       % display the first 5 columns
[~, Ypred] = max(Ypred);            % find the indices of max probabilities
sum(Ytest == Ypred) / length(Ytest) % compare the predicted vs. actual

%% Network Architecture
% You probably noticed that the artificial neural network model generated
% from the Pattern Recognition Tool has only one hidden layer. You can
% build a custom model with more layers if you would like, but this simple
% architecture is sufficient for most common problems.
% 
% The next question you may ask is how I picked 100 for the number of
% hidden neurons. The general rule of thumb is to pick a number between the
% number of input neurons, 784 and the number of output neurons, 10, and I
% just picked 100 arbitrarily. That means you might do better if you try
% other values. Let's do this programmatically this time. |myNNscript.m|
% will be handy for this - you can simply adapt the script to do a
% parameter sweep.

sweep = [10,50:50:300];                 % parameter values to test
scores = zeros(length(sweep), 1);       % pre-allocation
models = cell(length(sweep), 1);        % pre-allocation
x = Xtrain;                             % inputs
t = Ytrain;                             % targets
trainFcn = 'trainscg';                  % scaled conjugate gradient
for i = 1:length(sweep)
    hiddenLayerSize = sweep(i);         % number of hidden layer neurons
    net = patternnet(hiddenLayerSize);  % pattern recognition network
    net.divideParam.trainRatio = 70/100;% 70% of data for training
    net.divideParam.valRatio = 15/100;  % 15% of data for validation
    net.divideParam.testRatio = 15/100; % 15% of data for testing
    net = train(net, x, t);             % train the network
    models{i} = net;                    % store the trained network
    p = net(Xtest);                     % predictions
    [~, p] = max(p);                    % predicted labels
    scores(i) = sum(Ytest == p) /...    % categorization accuracy
        length(Ytest);                
end

%% 
% Let's now plot how the categorization accuracy changes versus number of
% neurons in the hidden layer. 
figure
plot(sweep, scores, '.-')
xlabel('number of hidden neurons')
ylabel('categorization accuracy')
title('Number of hidden neurons vs. accuracy')

%%
% It looks like you get the best result around 250 neurons and
% the best score will be around 0.96 with this basic artificial neural
% network model.
%
% As you can see, you gain more accuracy if you increase the number of
% hidden neurons, but then the accuracy decreases at some point (your
% result may differ a bit due to random initialization of weights). As you
% increase the number of neurons, your model will be able to capture more
% features, but if you capture too many features, then you end up
% overfitting your model to the training data and it won't do well with
% unseen data. Let's examine the learned weights with 300 hidden neurons.
% You see more details, but you also see more noise.


%% The Next Step - an Autoencoder Example
% You now have some intuition on artificial neural networks - a network
% automatically learns the relevant features from the inputs and generates
% a sparse representation that maps to the output labels.  What if we use
% the inputs as the target values? That eliminates the need for training
% labels and turns this into an unsupervised learning algorithm. This is
% known as an autoencoder and this becomes a building block of a deep
% learning network. There is an excellent example of autoencoders on the
% <http://www.mathworks.com/help/nnet/examples/training-a-deep-neural-network-for-digit-classification.html
% Training a Deep Neural Network for Digit Classification> page in the
% Neural Network Toolbox documentation, which also uses MNIST dataset. For
% more details, Stanford provides an excellent
% <http://deeplearning.stanford.edu/tutorial/ UFLDL Tutorial> that also
% uses the same dataset and MATLAB-based starter code.
% 
%% Sudoku Solver: a Real-time Processing Example
% Beyond understanding the algorithms, there is also a practical question
% of how to generate the input data in the first place. Someone spent a lot
% of time to prepare the MNIST dataset to ensure uniform sizing, scaling,
% contrast, etc. To use the model you built from this dataset in practical
% applications, you have to be able to repeat the same set of processing on
% new data. How do you do such preparation yourself?
%
% There is a fun video that shows you how you can solve Sudoku puzzles
% using a webcam that uses a different character recognition technique.
% Instead of static images, our colleague
% <http://www.mathworks.com/matlabcentral/profile/authors/1905880-teja-muppirala
% Teja Muppirala> uses a live video feed in real time to do it and he walks
% you through the pre-processing steps one by one. You should definitely
% check it out:
% <http://www.mathworks.com/videos/solving-a-sudoku-puzzle-using-a-webcam-68773.html
% Solving a Sudoku Puzzle Using a Webcam>.
%
% <<solveSudokuWebcam.png>>

%% Submitting Your Entry to Kaggle
% You got 96% categorization accuracy rate by simply accepting the default
% settings except for the number of hidden neurons. Not bad for the first
% try. Since you are using a Kaggle dataset, you can now submit your result
% to Kaggle. 

n = size(sub, 1);                                   % num of samples
sub = sub';                                         % transpose
[~, highest] = max(scores);                         % highest scoring model
net = models{highest};                              % restore the model
Ypred = net(sub);                                   % label probabilities
[~, Label] = max(Ypred);                            % predicted labels
Label = Label';                                     % transpose Label
Label(Label == 10) = 0;                             % change '10' to '0'
ImageId = 1:n; ImageId = ImageId';                  % image ids
writetable(table(ImageId, Label), 'submission.csv');% write to csv

%% 
% You can now submit the |submission.csv| on
% <https://www.kaggle.com/c/digit-recognizer/submissions/attach Kaggle's
% entry submission page>.

%% Closing
% In this example we focused on getting a high level intuition on
% artificial neural network using a concrete example of handwritten digit
% recognition. We didn’t go into details such as how the inputs weights and
% bias units are combined, how activation works, how you train such a
% network, etc. But you now know enough to use Neural Network Toolbox in
% MATLAB to participate in a Kaggle competition.

%%
% _Copyright 2015 The MathWorks, Inc._