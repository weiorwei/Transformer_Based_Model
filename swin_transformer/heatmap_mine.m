heatmap(result1);

colormap('jet')


x=find(result1(100,:)>0);



% figure
% x=rand(50,50);
% heatmap(x,'title','表格型热力图50*50随机矩阵');
% colormap('cool')