close all
clear
disp_freq = 100;
max_epoch = 40000;
err_goal = 1e-40;%0.01;
max_fail = 10000;
load iris

S1=31;      %for ind_S1 % METAPARAMETR
S2=30;      %for ind_S2 % METAPARAMETR

Ptest = zeros([4,45]);
Plearn = zeros([4,105]);
Ttest = zeros([1,45]);
Tlearn = zeros([1,105]);
results = zeros([1,5]);
errors = zeros([1,5]);


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
    for lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
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
        u = net(Ptest);
        errors(i) = sse(net, Ttest, u);
        results(i) = results(i) + (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100;
        i = i + 1;
    end

results = results / 10;
errors = errors / 10;
LRrange = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

plot(LRrange,results);
title('Zależność poprawności klasyfikacji od współczynnika uczenia');
xlabel('współczynnik uczenia lr');
ylabel('poprawność klasyfikacji [%]');
grid;

plot(LRrange,errors);
title('Zależność błędu SSE od współczynnika uczenia');
xlabel('współczynnik uczenia lr');
ylabel('SSE [%]');
grid;