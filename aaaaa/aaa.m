close all
clear
disp_freq = 100;
max_epoch = 40000;
err_goal = 1e-40;%0.01;
max_fail = 10000;
load iris

S1=41;      %for ind_S1 % METAPARAMETR
S2=31;      %for ind_S2 % METAPARAMETR
lr = 1e-5;

Ptest = zeros([4,45]);
Plearn = zeros([4,105]);
Ttest = zeros([1,45]);
Tlearn = zeros([1,105]);
results = zeros([1,5]);
errors = zeros([1,5]);

for tries = 1:10
    i = 1;
    r = 1;
    t = 1;
    for o = 1:3
        [traind] = crossvalind('Holdout', 50, 0.8);
        for k = 1:50
           if (traind(k) == 1)
               Ptest(:,t) = Pn(:,k+50*(o-1));
               Ttest(1,t) = T(1,k+50*(o-1));
               t = t + 1;
           else
               Plearn(:,r) = Pn(:,k+50*(o-1));
               Tlearn(1,r) = T(1, k+50*(o-1));
               r = r + 1;
           end
        end
    end
net = feedforwardnet([S1, S2],'traingd');   %definicja percepton
net.trainParam.epochs = max_epoch;  %maksymalna liczba epok
net.trainParam.goal = err_goal; %cel wydajności
net.trainParam.lr = lr; %learning rate
net.trainParam.max_fail = max_fail; %maksymalna ilość błędów walidacji
net.trainParam.showWindow = false;  %czy pokazać okno uczenia
net.divideParam.trainRatio=1;   %ilość danych do uczenia
net.divideParam.valRatio=0; %ilość danych do walidacji
net.divideParam.testRatio=0;    %ilość danych do testowania
[net,tr] = train(net,Plearn,Tlearn); %uczenie sieci neuronowej
u = u + net(Ptest);
end
u = u/10;
plot([1:length(Ttest)],Ttest,'r',[1:length(Ttest)],u,'g');
grid