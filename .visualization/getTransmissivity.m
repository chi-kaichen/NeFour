function t=getTransmissivity(dark,A,I)
%���͸����
omega=0.85;%������������
r = 60;%�����ߴ�
eps = 10^-6;%˳����
dark_refined=imguidedfilter(dark,I,'NeighborhoodSize',r,'DegreeOfSmoothing',eps);
t0=1-omega*(dark./A);
%figure;
%subplot(121);
%imshow(t0);
%title('��͸����ͼ');%�����͸����ͼ
%imwrite(t0,'t0.jpg');
t=1-omega*(dark_refined./A);
%subplot(122);
%imshow(t);
%title('ϸ͸����ͼ');%���ϸ͸����ͼ
%imwrite(t,'t.jpg')
