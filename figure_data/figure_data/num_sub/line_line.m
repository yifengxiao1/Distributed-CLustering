function [] = line_line(fid, data, name, yRange, MS, LW, FS, legend_list, lineType_list, Xtick, Xlabel, N)
    colorMap = lines(N);
    colorMap = swap(colorMap, 3, 4);
    colorMap = colorMap*0.85;    
    figure(fid);
    hold on;
%     lineType = ['-o'; '-s';  '-+'; '-*'; '-^'];
    for i = 1:N
        plot(Xtick, data(:, i), lineType_list(i, :),  'MarkerSize', MS, 'LineWidth', LW, 'Color', colorMap(i, :));
    end
    title(name)
    if yRange(1) > -1e-6
        ylim(yRange);
    end
    set(gca,'xtick', Xtick);
    xlabel(Xlabel);
%     legend('k-means||', 'DBDC', 'LSHDDP', 'REMOLD', 'LDSDC');
    if size(legend_list, 2) > 5
        legend(legend_list, 'Location', 'northwest', 'NumColumns', 2);
    else
        legend(legend_list);
    end
    box on;
    grid on;
    set(gca,'FontSize', FS);     
    set(gcf,'unit','normalized','position',[0.2,0.2,0.5,0.66]);
%     cm = [
%     0 0 .5;
%     0 0.5 0;
%     .86 .08 .24;
%     .5 0 .5;
%     1 .55 0
%     ];
%     colormap(gca, cm);
end

function [ret] = swap(par, i, j)
    tmp = par(i, :);
    par(i, :) = par(j, :);
    par(j, :) = tmp;
    ret = par;
end
