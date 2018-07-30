function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
a2 = sigmoid(Theta1*X');%'
a2 = [ones(1,m); a2];
a3 = sigmoid(Theta2*a2);

for i = 1:m,
	ans = zeros(num_labels,1);
	ans(y(i)) = 1;
	h = ones(num_labels,1);
	h = a3(:,i);
	J = J - sum((ans.*log(h) + (1-ans).*log(1-h)));
	end;
J = J/m;
some = 0;
[a,b] = size(Theta1);
for i = 1:a,
	for j = 2:b,
		some = some + Theta1(i,j)^2;
		end;
	end;	
[a,b] = size(Theta2);
for i = 1:a,
	for j = 2:b,
		some = some + Theta2(i,j)^2;
		end;
	end;
J = J + some*(lambda)/2/m;

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
addterm1 = zeros(size(Theta1));
addterm2 = zeros(size(Theta2));
delta1 = zeros(input_layer_size + 1, 1);
delta2 = zeros(hidden_layer_size + 1,1);
delta3 = zeros(num_labels + 1,1);
A1 = X';%'
A2 = sigmoid(Theta1*A1);
A2 = [ones(1,m);A2];
A3 = sigmoid(Theta2*A2);
A3 = [ones(1,m);A3];
for i = 1:m,
	a1vec = A1(:,i);
	a2vec = A2(:,i);
	a3vec = A3(:,i);
	ans = zeros(num_labels ,1);
	ans(y(i)) = 1;
	ans = [1; ans];
	delta3 = a3vec - ans;
	delta2 = (Theta2'*delta3(2:end)).*a2vec.*(1-a2vec);
	delta1 = (Theta1'*delta2(2,end)).*a1vec.*(1-a1vec);
	Delta1 = Delta1 + delta2(2:end)*a1vec';%';
	Delta2 = Delta2 + delta3(2:end)*a2vec';%';
	end;

addterm1 = lambda * Theta1;
addterm2 = lambda * Theta2;
for i = 1:size(Theta1, 1),
	addterm1(i,1) = 0;
	end;
for i = 1:size(Theta2, 1),
	addterm2(i,1) = 0;
	end;
Theta1_grad = (1/m)*(Delta1 + addterm1);
Theta2_grad = (1/m)*(Delta2 + addterm2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
