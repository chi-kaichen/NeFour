function A=getIntensity(dark)
%在图像的暗通道中用直方图获得大气光
[counts,~]=imhist(dark); %[count, x] = imhist( i ) 获取直方图信息，count为每一级灰度像素个数，x为灰度级
[m,n]=size(dark);
N=m*n;
sum=N*0.0005;%最亮的像素占总数的0.5%
histsum=0;%像素数目
h=256;%灰度级从256开始计算
A_sum=0;%亮度总和
while histsum<sum
    histsum=histsum+counts(h);
    h=h-1;
end
%i=h;%save the value of h
while h<256
    A_sum=A_sum+counts(h+1)*(h/255);
    h=h+1;
end
A=A_sum/histsum;