function dark=getDarkChannel(I)

%获得图像的r,g,b三个通道
r=I(:,:,1);
g=I(:,:,2);
b=I(:,:,3);
%将RGB三个像素矩阵中对应位置的最小值赋给一个新的矩阵
[m,n]=size(r);%设置暗通道大小
a=zeros(m,n);%从初始位置开始
%将RGB三个像素矩阵中对应位置的最小值赋给一个新的矩阵
for i=1:m
    for j=1:n
        a(i,j)=min(r(i,j),g(i,j));%先比较r,g通道
        a(i,j)=min(a(i,j),b(i,j));%再与蓝色通道b比较最小的
    end
end
%取原图像每个5*5邻域内的最小值，赋给该邻域内所有像素
d=ones(2,2);
fun=@(block_struct)min(min(block_struct.data))*d;
dark=blockproc(a,[2,2],fun);

dark=dark(1:m,1:n);%重新确定暗通道图的大小（宽度和高度）