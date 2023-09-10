N = 8;
name = 'NMI';
my_opt = load(sprintf('%s.txt', name));
my_opt = my_opt(:, 1:N);
num = 1:N;
close all;
figure(1);
hold on;
    cm = [
    0 0 .5;
    0 0.5 0;
    .86 .08 .24;
    .5 0 .5;
    1 .55 0
    ];
    colormap(cm);
lineType = ['-o'; '-s'; '-^'];
for i = 1:3
    plot(num, my_opt(i,:), lineType(i, :),  'MarkerSize', 12, 'LineWidth', 1.3);
end
title(sprintf('optimal %s', upper(name)));
xlabel('Dataset: G-%d');
legend('REMOLD', 'REMOLD+PCA', 'REMOLD+CGM');
box on;
grid on;
set(gca,'FontSize',15);
