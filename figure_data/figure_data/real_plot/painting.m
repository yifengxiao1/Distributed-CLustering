close all;
N = 10;
dat = importdata('Real_result.txt');
dat_nmi = dat(1:4:20, :);
line_line_hist(1, dat_nmi, 'NMI', [0 1.1]);
dat_purity = dat(2:4:20, :);
line_line_hist(2, dat_purity, 'Purity', [0 1.1]);
dat_trans = dat(3:4:20, :);
log_line_hist(3, dat_trans, 'Transmission Cost (MB)', [0 2e6]);
dat_time = dat(4:4:20, :);
log_line_hist(4, dat_time, 'Time (s)', [0 2e5]);





