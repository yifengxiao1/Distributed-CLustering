N = 8;
trans = load('transmission.txt');
trans = trans(:, 1:N);
num = 1:N;
close all;
% figure(1);
% hold on;
lineType = ['-o'; '-s'; '-^'];
    cm = [
    0 0 .5;
    0 0.5 0;
    .86 .08 .24;
    .5 0 .5;
    1 .55 0
    ];
    colormap(cm);
for i = 1:3
    semilogy(num, trans(i,:), lineType(i, :), 'MarkerSize', 12, 'LineWidth', 1.3);
    hold on;
end
xlabel('Dataset: G-%d');
title('Transmission Volume (MB)');
legend('REMOLD', 'REMOLD+PCA', 'REMOLD+CGM');
grid on;
set(gca,'FontSize',15);
