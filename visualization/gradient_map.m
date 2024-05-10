clear all; 
close all;
clc
for i = 10:10
    P = imread([num2str(i,'%dwo LSS'),'.png']);
%    d=size(P); 
% if(length(d)==3)      %判断图像是彩色图还是灰度图
%     P=rgb2gray(P); 
% end
    P=rgb2gray(P);
    P=double(P);  
    [Px,Py]=gradient(P); 
    G=sqrt(20*Px.*Px + 20*Py.*Py); %对梯度值进行处理
    P=G;  
    P=uint8(P);  
    imwrite(P,[num2str(i,'%dwo LSStidu'),'.png']);
% tou=input('input threshold tou=');  
% bh=input('input a constant intensity nearer to white bh='); 
    tou=1;
    bh=255;
    P1=P;  
    k=find(G>=tou); 
    P1(k)=bh;  
%     subplot(224);
%     imshow(P1); 
%     title('梯度图像2');
%     imwrite(P1,[num2str(i,'%gGrad'),'.png']);
% bl=input('input a constant intensity nearer to black bl='); 
% P2=P;  
% h=find(G>=tou); 
% P2(h)=bh; 
% l=find(G<tou); 
% P2(l)=bl;  
% subplot(235);
% imshow(P2); 
% title('梯度图像3');   
% P3=zeros(d(1),d(2)); 
% for i=1:d(1)     
%     for j=1:d(2)          
%         if (G(i,j)>=tou)             
%             P3(i,j)=bh;         
%         else
%             P3(i,j)=bl;         
%         end
%     end
% end
% P3=uint8(P3);  
% subplot(236);
% imshow(P3);
end
