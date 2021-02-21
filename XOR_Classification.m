%%
clc;
clear all;

Iteration_Size = 20000;
Learning_Rate = 0.005;

%% XOR CLASSIFICATION
XOR_Input = [0 0; 0 1; 1 0; 1 1];
XOR_Output = [0; 1; 1; 0];

XOR_Input_Neurons = size(XOR_Input, 2);
XOR_Hidden_Neurons = 2;
XOR_Output_Neurons = 2;
XOR_Hidden_Layer = 1;

Sample_Size = size(XOR_Input,1);

XOR_W1 = randn(XOR_Input_Neurons, XOR_Hidden_Neurons)/sqrt(XOR_Input_Neurons);
XOR_W2 = randn(XOR_Hidden_Neurons, XOR_Output_Neurons)/sqrt(XOR_Output_Neurons);

XOR_B1 = zeros(1,XOR_Hidden_Neurons)+0.01;
XOR_B2 = zeros(1,XOR_Output_Neurons)+0.01;

XOR_Y = [XOR_Output ~XOR_Output];

%%
for i=1:Iteration_Size
    
    XOR_Z1 = (XOR_Input * XOR_W1) + XOR_B1;
    XOR_A1 = max(0,XOR_Z1);
    
    XOR_Z2 = (XOR_A1 * XOR_W2) + XOR_B2;
    XOR_A2 = exp(XOR_Z2)./sum(exp(XOR_Z2),2);
    
    XOR_D1 = XOR_A2 - XOR_Y;
    XOR_DW1 = (XOR_A1')*(XOR_D1);
    XOR_DB1 = sum(XOR_D1,1);
    
    XOR_D2 = (XOR_D1*(XOR_W2')) .* (1*(XOR_Z1>=0));
    XOR_DW2 = (XOR_Input')*XOR_D2;
    XOR_DB2 = sum(XOR_D2,1);

    XOR_W1 = XOR_W1-Learning_Rate*XOR_DW2;
    XOR_W2 = XOR_W2-Learning_Rate*XOR_DW1;
    
    XOR_B1 = XOR_B1-Learning_Rate*XOR_DB2;
    XOR_B2 = XOR_B2-Learning_Rate*XOR_DB1;
end

%% XOR Plot
figure(1)
[Z,Y] = meshgrid(-2:0.1:2,-2:0.1:2);
Z_New = [Z(:) Y(:)];
XOR_Z1 = Z_New*XOR_W1+XOR_B1;
XOR_A1 = max(0,XOR_Z1);
XOR_Z2 = XOR_A1*XOR_W2+XOR_B2;
XOR_A2 = exp(XOR_Z2)./sum(exp(XOR_Z2),2);
[m,n]=max(XOR_A2,[],2);
XOR_Y = ~(n-1);
XOR_Y = reshape(XOR_Y,size(Z));
surf(Z,Y,double(XOR_Y));
title('XOR Classification')

%% REGRESSION
for j = [3 20]
    
rng(100);
X = (2*rand(1,50)-1)';
T = (sin(2*pi*X')+0.3*randn(1,50))';

Hidden_layer = 1;
Hidden_Neurons = j;
Input_Neurons = size(X,2);
Output_Neurons = 1;

Sample = size(X,1);
Rel_Factor = 0;

W1 = randn(Input_Neurons,Hidden_Neurons)/sqrt(Input_Neurons);
W2 = randn(Hidden_Neurons,Output_Neurons)/sqrt(Output_Neurons);

B1 = zeros(1,Hidden_Neurons) + 0.01;
B2 = zeros(1,Output_Neurons) + 0.01;

%%
for i=1:Iteration_Size
    Z1 = (X* W1) + B1;
    A1 = tanh(Z1);
    Z2 = (A1*W2) + B2;
    A2 = Z2;
    D1 = A2 - T;
    DW1 = (A1')*(D1);
    DB1 = sum(D1,1);
    D2 = (D1*(W2')) .* (1-A1.^2);
    DW2 = (X')*D2;
    DB2 = sum(D2,1);
    W1 = W1-Learning_Rate*DW2;
    W2 = W2-Learning_Rate*DW1;
    B1 = B1-Learning_Rate*DB2;
    B2 = B2-Learning_Rate*DB1;
end
%% Regression Plot 
figure(2)
plot(X,T,'rs')
hold on
[~,idx] = sort(X(:,1));
sortedmat = [X(idx,:) A2(idx,:)]; 
plot(sortedmat(:,1),sortedmat(:,2),'b-');
grid on;
legend('3-NA','3-NA','20-NA','20-NA') %NA - Network Architecture
title('Regression')
end
