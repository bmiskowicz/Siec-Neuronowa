%wyczyszczenia środowiska i ustawienie wartości początkowych
close all
clear
disp_freq = 100;
max_epoch = 40000;
max_fail = 10000;
load iris %wczytanie zbioru
%utworzenie potrzebnych tablic na dane i wartości
Ptest = zeros([4,30]);
Plearn = zeros([4,120]);
Ttest = zeros([1,30]);
Tlearn = zeros([1,120]);
results = zeros([11, 11]);
errors = zeros([11, 11]);

err_goal = 0.25/length(Plearn);%obliczenie pożądanej wartości błędu
j = 0;  %j to zmienna odpowiadająca za pokazywanie prograsu uczenia
lr = 1e-1;  %współczynnik uczenia, zmienian dla kolejnych eksperymentów

for tries = 1:10 %dla 10 powtórzeń
    r = 1;
    t = 1;
    for o = 1:3 %podział zbioru na klasy
        [traind] = crossvalind('Holdout', 50, 0.8);    %kroswalidacja danych
        for k = 1:50    %w każdej klasie dane są podzielone na uczące i testujące
           if (traind(k) == 1) %jeżeli indeks jest równy jeden
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
    for S1_vec = 1:10:101   %iteracja po pierwszej warstwie neuronów
        for S2_vec = 1:10:S1_vec    %iteracja po drugiej warstwie neuronów
            j = j + 1;  %zwiększanie progresu uczenia
            net = feedforwardnet([S1_vec, S2_vec],'traingd');   %definicja percepton
            net.trainParam.epochs = max_epoch;  %maksymalna liczba epok
            net.trainParam.goal = err_goal; %cel wydajności
            net.trainParam.lr = lr; %learning rate
            net.trainParam.max_fail = max_fail; %maksymalna ilość błędów walidacji
            net.trainParam.showWindow = false;  %czy pokazać okno uczenia
            net.divideParam.trainRatio=1;   %ilość danych do uczenia
            net.divideParam.valRatio=0; %ilość danych do walidacji
            net.divideParam.testRatio=0;    %ilość danych do testowania
            [net,tr] = train(net,Plearn,Tlearn); %uczenie sieci neuronowej
            u = net(Ptest); %zapisanie do u tablicy otrzymanych wyjść
            error = sse(net, Ttest, u); %olbiczenie SSE
            pk = (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100; %obliczenie PK
            results((S1_vec+9)/10, (S2_vec+9)/10) = results((S1_vec+9)/10, (S2_vec+9)/10) + pk;   %zapisanie do result PK
            errors((S1_vec+9)/10, (S2_vec+9)/10) = errors((S1_vec+9)/10, (S2_vec+9)/10) + error;  %zapisanie do errors SSE
            process = j/6.6;%obliczenie progresu
            fprintf('progress: %.2f%%\n', process);%wyświetlenie progresu
        end
    end
end

%wypełnienie wartościami 'NaN' pustych miejsc w tablicy (bo S2<S1)
for S2_vec = 1:10:101
    for S1_vec = 1:10:S2_vec
        results((S1_vec+9)/10, (S2_vec+9)/10) = NaN;
        errors((S1_vec+9)/10, (S2_vec+9)/10) = NaN;
    end
end

%obliczenie średnich wyników
results = results / 10;
errors = errors / 10;
SL1 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101];
SL2 = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101];

%wyrysowanie wykresu pk
figure(1);
surf(SL1, SL2, results');
title('Zależność poprawności klasyfikacji od ilości nauronów na poszczególnych warstwach');
xlabel('liczba neuronów w warstwie 1');
ylabel('liczba neuronów w warstwie 2');
zlabel('poprawność klasyfikacji [%]');
grid;

%wyrysowanie wykresu sse
figure(2);
surf(SL1, SL2, errors');
title('Zależność błędu SSE od ilości nauronów w poszczególnych warstwach');
xlabel('liczba neuronów w warstwie 1');
ylabel('liczba neuronów w warstwie 2');
zlabel('SSE [%]');
grid;