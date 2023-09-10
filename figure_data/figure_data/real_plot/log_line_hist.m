function [] = log_line_hist(fid, data, name, yRange)
    figure(fid)
    bar(data, .9);
%     for i = 1:3
%         margin = 0.225;
%         shift = -0.115;
%         up = 0.03;
%         v = [i-margin i i+margin]+shift;
%         for j = 1:3
%             t = text(v(j), my_dat(i, j)+up, sprintf('%.2f', my_dat(i, j)));
%             t.FontSize = 12;
%         end
%     end
    % [n,y] = hist(my_dat);
    % bar(y,n);
    % text(y,n+0.5,num2str(n'));
    grid on;
    %ylim(yRange);
    title(name)
    set(gca,'XTickLabel',{'Salinas','PaviaU','PaviaC','MNIST', 'MNIST1M'});
    xlabel('Dataset');
    legend('k-means||','DBDC', 'LSHDDP', 'REMOLD', 'LDSDC');
    set(gca,'FontSize',15);  
    set(gca,'YScale','log')
    set(gcf,'unit','normalized','position',[0.2,0.2,0.5,0.68]);
end
