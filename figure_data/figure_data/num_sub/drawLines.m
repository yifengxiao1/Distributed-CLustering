

file_list = {'300000_64_120_G-64.txt', '350000_128_140_G-128.txt', '96000_128_3_normal_surface-8_128.txt', '192000_256_3_normal_surface-9_256.txt', '111104_206_16salinas_corrected_withXY_zscore_all.txt', '207400_105_9paviaU_withXY_zscore_all.txt', '60000_784_10_MNIST_zscore.txt', '783640_104_9pavia_withXY_zscore_all.txt'};
len = length(file_list);
len = 7;
data_array = cell(1, len);

for i=1:len
    data_array{i} = load(file_list{i});
end
data_mat = randn(4, size(data_array{i}, 1), len);

for i = 1:len
    for j = 1:4
        data_mat(j, :, i) = data_array{i}(:, j);
    end
end

close all;
MS = 12;  %MarkerSize
LW = 2.25;   %LineWidth  
FS = 16;  %FontSize
lengend_list = {'G-6', 'G-7', 'S-7', 'S-8', 'Salinas', 'PaviaU', 'MNIST', 'PaviaC'};
lineType_list = ['-h'; '-*';  '-o'; '-+'; '-^'; '-p'; '-x'; '-d'];
Xtick = 2:2:18; 
Xlabel = 'No. of Sub-sites';
title_list = {'NMI'; 'Purity'; 'Transmission Cost (MB)'; 'Time (s)'};

for i = 1:2
    line_line(i, squeeze(data_mat(i, :, :)), title_list{i}, [0 1.1], MS, LW, FS, lengend_list, lineType_list, Xtick, Xlabel, len);
end

for i = 3:4
    line_line(i, squeeze(data_mat(i, :, :)), title_list{i}, -1, MS, LW, FS, lengend_list, lineType_list, Xtick, Xlabel, len);
end









