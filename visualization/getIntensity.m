function A=getIntensity(dark)
%��ͼ��İ�ͨ������ֱ��ͼ��ô�����
[counts,~]=imhist(dark); %[count, x] = imhist( i ) ��ȡֱ��ͼ��Ϣ��countΪÿһ���Ҷ����ظ�����xΪ�Ҷȼ�
[m,n]=size(dark);
N=m*n;
sum=N*0.0005;%����������ռ������0.5%
histsum=0;%������Ŀ
h=256;%�Ҷȼ���256��ʼ����
A_sum=0;%�����ܺ�
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