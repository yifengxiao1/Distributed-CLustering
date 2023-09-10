name = 'purity mean';
my_dat = load(sprintf('%s.txt', name));
my_dat = my_dat(:, 9:11)';
close all;
bar(my_dat, .9);

for i = 1:3
    margin = 0.225;
    shift = -0.115;
    up = 0.03;
    v = [i-margin i i+margin]+shift;
    for j = 1:3
        t = text(v(j), my_dat(i, j)+up, sprintf('%.2f', my_dat(i, j)));
        t.FontSize = 12;
    end
end
% [n,y] = hist(my_dat);
% bar(y,n);
% text(y,n+0.5,num2str(n'));
grid on;
ylim([0, 1.1]);
if strcmp(name, 'NMI')
    title('Optimal NMI');
end
if strcmp(name, 'nmi mean')
    title('Average NMI');
end
if strcmp(name, 'purity mean')
    title('Average Purity');
end
if strcmp(name, 'trans')
    title('Transmission Volume(MB)');
end
set(gca,'XTickLabel',{'Salinas','PaviaU','PaviaC'});
xlabel('Dataset');
legend('REMOLD','REMOLD+PCA', 'REMOLD+CGM');
set(gca,'FontSize',15);

