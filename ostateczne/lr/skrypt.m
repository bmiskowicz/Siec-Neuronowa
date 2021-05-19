result1 = 0;
result2 = 0;
result3 = 0;
result4 = 0;
result5 = 0;
error1 = 0;
error2 = 0;
error3 = 0;
error4 = 0;
error5 = 0;
for x = 2:1:11
    for y = 1:1:x-1
        result1 = result1 + results1(x, y);
        result2 = result2 + results2(x, y);
        result3 = result3 + results3(x, y);
        result4 = result4 + results4(x, y);
        result5 = result5 + results5(x, y);
        error1 = error1 + errors1(x, y);
        error2 = error2 + errors2(x, y);
        error3 = error3 + errors3(x, y);
        error4 = error4 + errors4(x, y);
        error5 = error5 + errors5(x, y);
        
    end
end

result1 = result1 / 55;
result2 = result2 / 55;
result3 = result3 / 55;
result4 = result4 / 55;
result5 = result5 / 55;
error1 = error1 / 55;
error2 = error2 / 55;
error3 = error3 / 55;
error4 = error4 / 55;
error5 = error5 / 55;

range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
results = [result5, result4, result3, result2, result1];
errors = [error5, error4, error3, error2, error1];

figure(1);
plot(range,results);
title('Zależność poprawności klasyfikacji od współczynnika uczenia');
xlabel('współczynnik uczenia lr');
ylabel('poprawność klasyfikacji [%]');
grid;

figure(2);
plot(range,errors);
title('Zależność błędu SSE od współczynnika uczenia');
xlabel('współczynnik uczenia lr');
ylabel('SSE');
grid;