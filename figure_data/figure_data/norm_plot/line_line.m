function [] = line_line(fid, data, name, yRange, MS, LW, FS)
    figure(fid);
    darker = 0.85;
    hold on;
    lineType = ['-o'; '-s';  '-+'; '-*'; '-^'];
    colorMap = [0, 114, 189; 217,83,25; 237,177,32; 126,47,142; 119,172,48];
    colorMap(3, :) = colorMap(3, :)*darker;
    colorMap(5, :) = colorMap(5, :)*darker;
    for i = 1:5
        plot(1:10, data(:, i), lineType(i, :),  'MarkerSize', MS, 'LineWidth', LW, 'Color', colorMap(i,:)/255);
    end
    title(name)
    ylim(yRange)
    xlabel('Index of Dataset G');
    legend('k-means||', 'DBDC', 'LSHDDP', 'REMOLD', 'LDSDC');
    box on;
    grid on;
    set(gca,'FontSize', FS);     
    set(gcf,'unit','normalized','position',[0.2,0.2,0.5,0.66]);
end