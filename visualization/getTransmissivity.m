function t=getTransmissivity(dark,A,I)
%获得透射率
omega=0.85;%雾气保留因子
r = 60;%邻区尺寸
eps = 10^-6;%顺滑度
dark_refined=imguidedfilter(dark,I,'NeighborhoodSize',r,'DegreeOfSmoothing',eps);
t0=1-omega*(dark./A);
%figure;
%subplot(121);
%imshow(t0);
%title('粗透射率图');%输出粗透射率图
%imwrite(t0,'t0.jpg');
t=1-omega*(dark_refined./A);
%subplot(122);
%imshow(t);
%title('细透射率图');%输出细透射率图
%imwrite(t,'t.jpg')
