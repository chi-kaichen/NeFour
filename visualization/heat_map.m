clc
clear all;
close all;

for i = 1:890
    f = imread([num2str(i,'%d [clahe] [lab]'),'.png']);
    I = double(f)/255;
    dark = getDarkChannel(I);%��ð�ͨ��ͼ
    A = getIntensity(dark);%��������ǿ
    t = getTransmissivity(dark,A,I);%���͸����
    imwrite(t,[num2str(i,'%dtm'),'.png']);
end

% ��ͼ
for i = 1:890
    img = imread([num2str(i,'%dtm'),'.png']);
%     img = double(f)/255;
%     img = rgb2gray(f);
    [m,n] = size(img);
    set (gcf,'Position',[0,0,n,m])
    imshow(img,'border','tight','initialmagnification','fit');
    colormap(jet);
    F = getframe(gcf);
    imwrite(F.cdata,[num2str(i,'%d [clahe] [lab]retu'),'.png'])
end