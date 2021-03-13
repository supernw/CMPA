clear;

%Constants
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

I = @(V) Is.*(exp((1.2/0.025).*V) - 1) + Gp.*V - Ib.*(exp((-1.2/0.025).*(V+Vb))-1);
V_var = linspace(-1.95,0.7,200);

I_vec = I(V_var);
I_vec_rand = zeros(1,length(V_var));
for i=1:length(V_var)
    rand_p = rand()*0.2 +1; %Percentage 0%-20%
    rand_pm = randi(2,1)-1;
    
    if(rand_pm == 1) %+
        I_vec_rand(i) = I_vec(i) + I_vec(i)*rand_p;
    else
        I_vec_rand(i) = I_vec(i) - I_vec(i)*rand_p;
    end
end

%Polyfits
x_poly = linspace(-1.95,0.7);

p8 = polyfit(V_var,I_vec,8);
yp8 = polyval(p8,x_poly);

p4 = polyfit(V_var,I_vec,4);
yp4 = polyval(p4,x_poly);

p8_rand = polyfit(V_var,I_vec_rand,8);
yp8_rand = polyval(p8_rand,x_poly);

p4_rand = polyfit(V_var,I_vec_rand,4);
yp4_rand = polyval(p4_rand,x_poly);

%Plots
figure(1)
plot(V_var, I_vec)
hold on
plot(x_poly, yp8, '--');
plot(x_poly, yp4, '--');
hold off
title('Diode Current - No Noise')
xlabel('Voltage')
ylabel('Current')

figure(2)
semilogy(V_var,abs(I_vec))
hold on
semilogy(x_poly, abs(yp8), '--');
semilogy(x_poly, abs(yp4), '--');
hold off
title('Diode Current (LogScale) - No Noise')
xlabel('Voltage')
ylabel('Abs(Current)')

figure(3)
plot(V_var, I_vec_rand)
hold on
plot(x_poly, yp8_rand, '--');
plot(x_poly, yp4_rand, '--');
hold off
title('Diode Current - With Noise')
xlabel('Voltage')
ylabel('Current')


figure(4)
semilogy(V_var,abs(I_vec_rand))
hold on
semilogy(x_poly, abs(yp8_rand), '--');
semilogy(x_poly, abs(yp4_rand), '--');
hold off
title('Diode Current (LogScale) - No Noise')
xlabel('Voltage')
ylabel('Abs(Current)')

%Conclusions: This type of linear fit does not work well for log plots
%Question 2
fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
fo4 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');

%Fit 2
ff2 = fit(V_var',I_vec',fo2);
ff3 = fit(V_var',I_vec',fo3);
ff4 = fit(V_var',I_vec',fo4);

If2 = ff2(V_var');
If3 = ff3(V_var');
If4 = ff4(V_var');

figure(5)
plot(V_var,I_vec);
hold on
plot(V_var, If2', '--');
plot(V_var, If3', '--');
plot(V_var, If4', '--');
hold off
legend('Data', 'Fit2', 'Fit3', 'Fit4');
ylim([-4 4])
title('Diode Current  - No Noise - NonLinear Fits')
xlabel('Voltage')
ylabel('Abs(Current)')

figure(6)
semilogy(V_var,abs(I_vec));
hold on
semilogy(V_var, abs(If2'), '--');
semilogy(V_var, abs(If3'), '--');
semilogy(V_var, abs(If4'), '--');
hold off
legend('Data', 'Fit2', 'Fit3', 'Fit4');
title('Diode Current  - With Noise - NonLinear Fits')
xlabel('Voltage')
ylabel('Abs(Current)')

%Neural Net Fits
inputs = V_var'.';
targets = I_vec'.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs


