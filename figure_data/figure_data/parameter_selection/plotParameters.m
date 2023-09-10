close all;
strLegend = {'G-1', 'G-2', 'G-3', 'G-4', 'G-5', 'G-6', 'G-7', 'G-8', 'G-9', 'G-10'};
draw_it(1, 'Gaussian', [0 1.1], strLegend, 'Gaussian_parameters.txt', 10);
strLegend = {'S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-10'};
draw_it(2, 'Norm-Ball Surface', [0 1.1], strLegend, 'norm_parameters.txt', 10);
strLegend = {'Salinas', 'PaviaU', 'PaviaC', 'MNIST', 'MNIST1M'};
draw_it(3, 'Real-World', [0 1.1], strLegend, 'real_parameters.txt', 5);
