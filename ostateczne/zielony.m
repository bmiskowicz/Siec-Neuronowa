%wyczyszczenia środowiska i ustawienie wartości początkowych
close all
clear
disp_freq = 100;
max_epoch = 40000;
max_fail = 10000;
load iris
%ustawienie optymalnej ilości neuronów
S1=36;
S2=21;
lr = 0.1;

%stworzenie potrzebnych tablic
Ptest = zeros([4,30]);
Plearn = zeros([4,120]);
Ttest = zeros([1,30]);
Tlearn = zeros([1,120]);
err_goal = 0.25/length(Plearn);%błąd docelowy

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

net = feedforwardnet([S1, S2],'traingd');   %definicja percepton
net.trainParam.epochs = max_epoch;  %maksymalna liczba epok
net.trainParam.goal = err_goal; %cel wydajności
net.trainParam.lr = lr; %learning rate
net.trainParam.max_fail = max_fail; %maksymalna ilość błędów walidacji
net.trainParam.showWindow = true;  %czy pokazać okno uczenia
net.divideParam.trainRatio=1;   %ilość danych do uczenia
net.divideParam.valRatio=0; %ilość danych do walidacji
net.divideParam.testRatio=0;    %ilość danych do testowania
[net,tr] = train(net,Plearn,Tlearn); %uczenie sieci neuronowej
u = net(Ptest); %zapisanie do u tablicy otrzymanych wyjść
pk = (1-sum(abs(Ttest-u)>=0.5)/length(Ttest))*100; %obliczenie PK
plot((1:length(Ttest)),Ttest,'r',(1:length(Ttest)),u,'g');
grid;
