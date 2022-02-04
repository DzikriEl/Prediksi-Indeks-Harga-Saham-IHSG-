% TUGAS AKHIR OPTIMASI
% 065118123 - DZIKRI FAIZZIYAN
% 065118137 - M.FAJAR IKHSAN JA'FAR
% 065118180 - ANANDA REYNATA SAPUTRA

clc; clear; close all; warning off all;

% MEMBACA DATA EXCEL
data = xlsread('IHSG.xlsx',1,'F10:Q14');

% PROSES NORMALISASI DATA
max_data = max(max(data));
min_data = min(min(data));
 
[m,n] = size(data);
data_norm = zeros(m,n);
for x = 1:m
    for y = 1:n
        data_norm(x,y) = 0.1+0.8*(data(x,y)-min_data)/(max_data-min_data);
    end
end

% MENYUSUN DATA DAN TARGET LATIH
data_norm = data_norm';
tahun_latih = 3;	% Januari 2015 s.d November 2018
jumlah_bulan = 12;
data_latih = zeros(12,jumlah_bulan*tahun_latih);

for n = 1:jumlah_bulan*tahun_latih
    for m = 1:jumlah_bulan
        data_latih(m,n) = data_norm(m+n-1);
    end
end
 
target_latih = data_norm(jumlah_bulan+1:jumlah_bulan*(tahun_latih+1)); % Januari 2016 s.d Desember 2018

% menyiapkan parameter2 arsitektur jst (jumlah neuron pada
% hidden layer, jenis fungsi aktivasi, dan fungsi pelatihan)
jumlah_neuron1 = 10;
fungsi_aktivasi1 = 'logsig';
fungsi_aktivasi2 = 'purelin';
fungsi_pelatihan = 'traingd';

% membangun arsitektur jaringan syaraf tiruan backpropagation
net = newff(minmax(data_latih),[jumlah_neuron1 1],{fungsi_aktivasi1,...
    fungsi_aktivasi2},fungsi_pelatihan);

% menyiapkan parameter2 pelatihan (error goal, jumlah
% epoch, laju pembelajaran)
error_goal = 1e-6;
jumlah_epoch = 1000;
laju_pembelajaran = 0.01;

net.trainParam.goal = error_goal;
net.trainParam.epochs = jumlah_epoch;
net.trainParam.lr = laju_pembelajaran;

% proses pelatihan (training)
net_keluaran = train(net,data_latih,target_latih);

% hasil pelatihan
hasil_latih = sim(net_keluaran,data_latih);

% penghitungan nilai MSE
MSE_latih = mse(hasil_latih,target_latih);

hasil_latih = ((hasil_latih-0.1)*(max_data-min_data)/0.8)+min_data;
target_latih_asli = data(2:4,:);	% 2016 s.d 2018
target_latih_asli = target_latih_asli';
target_latih_asli = target_latih_asli(:);

% PSO
h = @(x) NMSE(x, net_keluaran, data_latih, target_latih);
k = jumlah_neuron1;
[x_pso, ~] = pso(h, 12*k+k+k+1);
net_keluaran_pso = setwb(net_keluaran, x_pso');
hasil_latih_pso = sim(net_keluaran_pso,data_latih);

% penghitungan nilai MSE
MSE_latih_pso = mse(hasil_latih_pso,target_latih);

hasil_latih_pso = ((hasil_latih_pso-...
    0.1)*(max_data-min_data)/0.8)+min_data;

% plot grafik keluaran jst dengan target
figure,
plot(target_latih_asli,'ro-','LineWidth',1)
hold on
plot(hasil_latih,'b*-','LineWidth',1)
plot(hasil_latih_pso,'g*-','LineWidth',1)
hold off
grid on
title('Grafik Keluaran JST vs Target')
h = gca;
h.XTick = [1:12:48];
h.XTickLabel = {'2016';'2017';'2018';'2019'};
xlabel('Tahun')
ylabel('IHSG')
legend('Target','NN','NN+PSO','Location','Best')

% menyusun data dan target uji
tahun_uji = 1; % Januari 2018 s.d November 2019
jumlah_bulan = 12;
data_uji = zeros(12,jumlah_bulan*tahun_uji-1);

for n = 1:jumlah_bulan*tahun_uji
    for m = 1:jumlah_bulan
        data_uji(m,n) = data_norm(jumlah_bulan*tahun_latih+m+n-1);
    end
end
 
target_uji = data_norm(jumlah_bulan*(tahun_latih+tahun_uji)+1:end); % Januari 2019 s.d Desember 2019

% hasil pengujian NN
hasil_uji = sim(net_keluaran,data_uji);

% penghitungan nilai MSE
MSE_uji = mse(hasil_uji,target_uji);

hasil_uji = ((hasil_uji-...
    0.1)*(max_data-min_data)/0.8)+min_data;
target_uji_asli = data(5,:);	% 2019
target_uji_asli = target_uji_asli';
target_uji_asli = target_uji_asli(:);

% hasil pengujian NN+PSO
hasil_uji_pso = sim(net_keluaran_pso,data_uji);

% error MSE PSO optimized NN
MSE_uji_pso = mse(hasil_uji_pso,target_uji);

hasil_uji_pso = ((hasil_uji_pso-...
    0.1)*(max_data-min_data)/0.8)+min_data;

% plot grafik keluaran jst dengan target
figure,
plot(target_uji_asli,'mo-','LineWidth',1)
hold on
plot(hasil_uji,'y*-','LineWidth',1)
plot(hasil_uji_pso,'c*-','LineWidth',1)
hold off
grid on
title('Grafik Keluaran JST vs Target')
h = gca;
h.XTick = [1:12];
h.XTickLabel = {'JAN';'FEB';'MAR';'APR';'MEI';'JUN';...
    'JUL';'AGS';'SEP';'OKT';'NOV';'DES'};
xlabel('Tahun 2019')
ylabel('IHSG')
legend('Target','NN','NN+PSO','Location','Best')