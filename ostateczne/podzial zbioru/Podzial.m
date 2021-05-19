%wyczyszczenia środowiska i ustawienie wartości początkowych
close all
clear
disp_freq = 100;
max_epoch = 40000;
max_fail = 10000;
load iris %wczytanie zbioru

%początkowe parametry ustawione są domyślnie
S1=10;  %ilość neuronów na pierwszej warstwie
S2=9;   %ilość neuronów na drugiej warstwie
lr = 0.1; %współczynnik uczenia

averages = zeros([1,9]);    %pusta tablica dla zmiennych

for tries = 1:1:10  %dla 10 powtórzeń
    for x = 1:9 %zbiór podzielony jest na 9 sposobów
        r = 1;
        t = 1;
        for o = 1:3 %podział zbioru na klasy
            [traind] = crossvalind('Holdout', 50, x/10);    %kroswalidacja danych
            for k = 1:50    %w każdej klasie dane są podzielone na uczące i testujące
               if (traind(k) == 1)  %jeżeli indeks jest równy jeden
                   %dana jest zapisywana do zbioru testującego
                   Ptest(:,t) = Pn(:,k+50*(o-1));
                   Ttest(1,t) = T(1,k+50*(o-1));
                   t = t + 1;   %iterowanie indeksu
               else
                   %w przeciwnym wypadku jest zapisywana do zbioru uczącego
                   Plearn(:,r) = Pn(:,k+50*(o-1));
                   Tlearn(1,r) = T(1, k+50*(o-1));
                   r = r + 1;   %iterowanie indeksu
               end
            end
        end
        net = feedforwardnet([S1, S2],'traingd');   %definicja perceptonu
        net.trainParam.epochs = max_epoch;  %maksymalna liczba epok
        net.trainParam.goal = 0.25/length(Ptest); %cel wydajności
        net.trainParam.lr = lr; %learning rate
        net.trainParam.max_fail = max_fail; %maksymalna ilość błędów walidacji
        net.trainParam.showWindow = false;  %czy pokazać okno uczenia
        net.divideParam.trainRatio=1;   %ilość danych do uczenia
        net.divideParam.valRatio=0; %ilość danych do walidacji
        net.divideParam.testRatio=0;    %ilość danych do testowania
        [net,tr] = train(net,Plearn,Tlearn); %uczenie sieci neuronowej
        u = net(Ptest); %zapisanie do u tablicy otrzymanych wyjść
        result = (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100;  %zapisanie do result poprawności klasyfikacji
        averages(x) = averages(x)+result;   %dodawanie PK dla kolejnycych powtórzeń
        clear Plearn; clear Ptest; clear Tlearn; clear Ttest; %zresetowanie tablic
    end
end
averages = averages/10; %wyciągnięcie  średniej PK
%rysowanie wykresu
percents = [10, 20, 30, 40, 50, 60, 70, 80, 90];
plot(percents, averages);
title('Zależność poprawności klasyfikacji od podziału zbioru');
xlabel('wielkość zbioru uczącego [% całości zbioru]') ;
ylabel('poprawność klasyfikacji [%]');
grid;