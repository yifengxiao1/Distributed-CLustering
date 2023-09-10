close all;
N = 10;
MS = 14;  %MarkerSize
LW = 2.25;   %LineWidth  
FS = 16;  %FontSize

dat = importdata('norm_result.txt');
dat_nmi = dat(1:4:40, :);
line_line(1, dat_nmi, 'NMI', [0 1.1], MS, LW, FS);
dat_purity = dat(2:4:40, :);
line_line(2, dat_purity, 'Purity', [0 1.1], MS, LW, FS);
dat_trans = dat(3:4:40, :);
log_line(3, dat_trans, 'Transmission Cost (MB)', MS, LW, FS);
dat_time = dat(4:4:40, :);
log_line(4, dat_time, 'Time (s)', MS, LW, FS);






