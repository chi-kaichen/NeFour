function dark=getDarkChannel(I)

%���ͼ���r,g,b����ͨ��
r=I(:,:,1);
g=I(:,:,2);
b=I(:,:,3);
%��RGB�������ؾ����ж�Ӧλ�õ���Сֵ����һ���µľ���
[m,n]=size(r);%���ð�ͨ����С
a=zeros(m,n);%�ӳ�ʼλ�ÿ�ʼ
%��RGB�������ؾ����ж�Ӧλ�õ���Сֵ����һ���µľ���
for i=1:m
    for j=1:n
        a(i,j)=min(r(i,j),g(i,j));%�ȱȽ�r,gͨ��
        a(i,j)=min(a(i,j),b(i,j));%������ɫͨ��b�Ƚ���С��
    end
end
%ȡԭͼ��ÿ��5*5�����ڵ���Сֵ����������������������
d=ones(2,2);
fun=@(block_struct)min(min(block_struct.data))*d;
dark=blockproc(a,[2,2],fun);

dark=dark(1:m,1:n);%����ȷ����ͨ��ͼ�Ĵ�С����Ⱥ͸߶ȣ�