close all
clear
disp_freq = 100;
max_epoch = 40000;
err_goal = 1e-40;%0.01;
max_fail = 10000;
load iris
lr = 0.1;   %for ind_lr % METAPARAMETR

Ptest = zeros([4,45]);
Plearn = zeros([4,105]);
Ttest = zeros([1,45]);
Tlearn = zeros([1,105]);
results = zeros([11, 11]);
errors = zeros([11, 11]);

for tries = 1:10
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
    for S1_vec = 1:10:101
        for S2_vec = 1:10:S1_vec
            net = feedforwardnet([S1_vec, S2_vec], 'traingd');   %definicja percepton
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
            error = sse(net, Ttest, u);
            results((S1_vec+9)/10, (S2_vec+9)/10) = results((S1_vec+9)/10, (S2_vec+9)/10) + (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100;
            errors((S1_vec+9)/10, (S2_vec+9)/10) = error;
        end
    end
end
results = results / 10;
errors = errors / 10;
SL1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101];
SL2 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101];
mesh(SL1, SL2, results');
title('Zależność poprawności klasyfikacji od ilości nauronów na poszczególnych warstwach');
xlabel('liczba neuronów w warstwie 1');
ylabel('liczba neuronów w warstwie 2');
zlabel('poprawność klasyfikacji [%]');
grid;

mesh(SL1, SL2, errors');
title('Zależność błędu SSE od ilości nauronów na poszczególnych warstwach');
xlabel('liczba neuronów w warstwie 1');
ylabel('liczba neuronów w warstwie 2');
zlabel('SSE [%]');
grid;