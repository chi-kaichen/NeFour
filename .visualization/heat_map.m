clc
clear all;
close all;

for i = 1:890
    f = imread([num2str(i,'%d [clahe] [lab]'),'.png']);
    I = double(f)/255;
    dark = getDarkChannel(I);%获得暗通道图
    A = getIntensity(dark);%求解大气光强
    t = getTransmissivity(dark,A,I);%求解透射率
    imwrite(t,[num2str(i,'%dtm'),'.png']);
end

% 热图
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