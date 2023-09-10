function [] = draw_it(fid, name, yRange, strLegend, fn, n)
%DRAW_IT 此处显示有关此函数的摘要
%   此处显示详细说明
    dat = importdata(fn);
    sampleN = 31;
    ms = 8;
    fs = 16;
    lw = 1.5;
    figure(fid);
    hold on;
    lineType = ['-o'; '-+'; '-*'; '-x'; '-s'; '-d'; '-^'; '-v'; '-p'; '-h'];
    colorMap = lines(n); 
    colorMap = colorMap*0.85;
    title(name);
    ylim(yRange);
    for i = 1:n
        sz = dat(i*2, 1);
        x = dat(i*2-1, 2:2+sz-1);
        y = dat(i*2, 2:2+sz-1);
        tsz = 1;
        for j = 2:sz
            if x(tsz) > x(j)
                tsz = tsz+1;
                x(tsz) = x(j);
                y(tsz) = y(j);
            end
        end
        xi = 0:1/(sampleN-1):1;
        yi = interp1(x(1:tsz), y(1:tsz), xi, 'method');
        plot(xi, yi, lineType(i, :),  'MarkerSize', ms, 'LineWidth', lw, 'Color', colorMap(i, :));
%        plot(x, y, 'MarkerSize', 12, 'LineWidth', 1.3);
    end
    box on;
    grid on;
    xlabel('$\delta$', 'Interpreter', 'latex');
    ylabel('NMI');
    set(gca,'FontSize',fs);     
    set(gcf,'unit','normalized','position',[0.2,0.2,0.55,0.66]);
    if size(strLegend, 2) > 5
        legend(strLegend, 'Location','northwest','NumColumns',2);
    else
        legend(strLegend);
    end
end

