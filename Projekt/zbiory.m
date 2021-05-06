close all
clear
disp_freq = 100;
max_epoch = 40000;
err_goal = 1e-40;%0.01;
max_fail = 10000;
load iris

S1=10;      %for ind_S1 % METAPARAMETR
S2=9;      %for ind_S2 % METAPARAMETR
lr = 0.1;   %for ind_lr % METAPARAMETR

averages = zeros([1,9]);

for tries = 1:1:20
    for x = 1:9
        r = 1;
        t = 1;
        for o = 1:3
            [traind] = crossvalind('Holdout', 50, x/10);
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
        net = feedforwardnet([S1, S2],'traingd');   %definicja perceptonu
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
        result = (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100;
        averages(x) = averages(x)+result;
        clear Plearn; clear Ptest; clear Tlearn; clear Ttest;
    end
end
percents = [10, 20, 30, 40, 50, 60, 70, 80, 90];
plot(percents, averages/20);
title('Zależność poprawności klasyfikacji od podziału zbioru');
xlabel('wielkość zbioru uczącego [% całości zbioru]') ;
ylabel('poprawność klasyfikacji [%]');
grid;