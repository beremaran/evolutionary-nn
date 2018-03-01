clc; clear;

data = importfile('fertility.csv');

X = data(:, 1:9);
y = data(:, 10);

X = normc(X);
for i = 1:size(X, 2)
    X(:, i) = X(:, i) + abs(min(X(:, i))) + (rand(size(X, 1), 1) / 100);
end

X = X';
y = full(ind2vec(y' + 1));

% =========================================================================

populasyon_buyuklugu = 10;
toplam_jenerasyonlar = 100;
minimum_skor = 0.95;

mutasyon_miktari = 5;
mutasyon_araligi = [-mutasyon_miktari, mutasyon_miktari];

maksimum_noronlar = 25;
maksimum_katmanlar = 5;

rng(0, 'twister');
noron_sayilari = randi([2, maksimum_noronlar], 1, populasyon_buyuklugu);
katman_sayilari = randi([2, maksimum_katmanlar], 1, populasyon_buyuklugu);

% 1 -> network
% 2 -> id
% 3- > skor
% 4 & 5 -> katman sayisi, noron sayisi
% 6 -> egitim kaydi
aglar = cell(populasyon_buyuklugu, 6);
fprintf("Ilk populasyon olusturuluyor ..\n");
for i = 1:populasyon_buyuklugu
    katmanlar = ones(1, katman_sayilari(i)) * noron_sayilari(i);
    ag = patternnet( katmanlar );

    ag = configure(ag, X, y);
    ag = init(ag);

    aglar{i, 1} = ag;
    aglar{i, 2} = i;
    aglar{i, 3} = 0;
    aglar{i, 4} = katman_sayilari(i);
    aglar{i, 5} = noron_sayilari(i);
end

fprintf("Genetik proses basliyor ..\n");
for j = 1:toplam_jenerasyonlar

    if aglar{1, 3} >= minimum_skor
        break
    end

    for i = 1:populasyon_buyuklugu
        katmanlar = ones(1, aglar{i, 4}) * aglar{i, 5};
        aglar{i, 1} = patternnet(katmanlar);
    end

    for i = 1:populasyon_buyuklugu

        aglar{i, 1}.trainFcn = 'trainscg';

        aglar{i, 1}.trainParam.showWindow = 0;

        aglar{i, 1} = configure(aglar{i, 1}, X, y);
        aglar{i, 1} = init(aglar{i});

        [aglar{i, 1}, aglar{i, 6}] = train(aglar{i, 1}, X, y);
        p = aglar{i, 1}(X);

        p(p < 0.5) = 0;
        p(p >= 0.5) = 1;

        aglar{i, 3} = rsquare(y, p);
    end

    aglar = sortrows(aglar, 3, 'descend');

    iyi_sayisi = round(populasyon_buyuklugu * 0.4);
    iyiler = aglar(1:iyi_sayisi, :);

    % 1 -> network
    % 2 -> id
    % 3- > skor
    % 4 & 5 -> katman sayisi, noron sayisi

    aglar_isaretci = size(aglar, 1);
    for i = 1:iyi_sayisi - 1
        b1 = iyiler(i, 4:5);
        b2 = iyiler(i + 1, 4:5);

        c1 = cell2mat([b1(1) b2(2)]);
        c2 = cell2mat([b2(1) b1(2)]);

        % 30% ihtimalle mutasyona u�rayabilirler
        c11 = c1(1);
        c12 = c1(2);

        c21 = c2(1);
        c22 = c2(2);

        if rand() > .5
            c1(1) = c1(1) + randi(mutasyon_araligi);
            c1(2) = c1(2) + randi(mutasyon_araligi);
        end

        if rand() > .5
            c2(1) = c2(1) + randi(mutasyon_araligi);
            c2(2) = c2(2) + randi(mutasyon_araligi);
        end

        if c1(1) <= 0
            c1(1) = c11;
        end

        if c1(2) <= 0
            c1(2) = c12;
        end

        if c2(1) <= 0
            c2(1) = c21;
        end

        if c2(2) <= 0
            c2(2) = c22;
        end

        aglar{aglar_isaretci, 4} = c1(1);
        aglar{aglar_isaretci, 5} = c1(2);

        aglar{aglar_isaretci - 1, 4} = c2(1);
        aglar{aglar_isaretci - 1, 5} = c2(2);

        aglar_isaretci = aglar_isaretci - 2;
    end

    fprintf("Jenerasyon #%03d | R2: %.2f | ", j, aglar{1, 3});
    fprintf("Katman sayisi: %03d - Noron sayilari: %03d\n", aglar{1, 4}, aglar{1,5});
    plotperform(aglar{1, 6})

end

fprintf("Genetik proses tamamlandi.\n");
% =========================================================================

en_iyi_ag = aglar{1, 1};

tahminler = en_iyi_ag(X);
tahminler = [ vec2ind(tahminler)', vec2ind(y)' ] - 1;

hatali_sayisi = 0;
for i = 1:length(tahminler)
    if tahminler(i, 1) ~= tahminler(i, 2)
        hatali_sayisi = hatali_sayisi + 1;
    end
end

fprintf("Hatali ornek sayisi: %03d/%03d (%03d%s)\n", hatali_sayisi, length(tahminler), (hatali_sayisi * 100 / length(tahminler) ), "%");

% =========================================================================

figure
scatter(tahminler(:, 2), 1:length(X), 100, 'magenta', 'filled')
hold on
scatter(tahminler(:, 1), 1:length(X), 50, 'green', 'filled')

xlim([-1, 2])
xlabel('Siniflar')

ylim([0, length(X) + 10])
ylabel('Ornek #')

grid on
legend('Hedefler', 'Tahminler')

figure
hold on
grid on

for i = 1:(round(populasyon_buyuklugu * 0.25))
    plot(aglar{i, 6}.vperf, ...
        'LineWidth', 2, ...
        'DisplayName', sprintf("#%d Validation", i), ...
        'Color', rand(1, 3));
    plot(aglar{i, 6}.tperf, ...
        'LineWidth', 2, ...
        'DisplayName', sprintf("#%d Test", i), ...
        'Color', rand(1, 3));
end

legend('show')

% =========================================================================

fprintf("Son populasyon:\n");

for i = 1:populasyon_buyuklugu
    a = aglar(i, :);

    p = a{1}(X);

    p(p < 0.5) = 0;
    p(p >= 0.5) = 1;

    fprintf("%02d\t%03d\t%03d\t%03.2f\n", a{2}, a{4}, a{5}, ...
        rsquare(y, p));
end
