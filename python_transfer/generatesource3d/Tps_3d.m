function Tps_3d()
vertices = load('vertices.mat');
original_p = load('original_point.mat');
target_p = load('target_point.mat');
vertices = vertices.vertices;
original_p = original_p.original_point;
target_p = target_p.target_point;
len_p = size(original_p,1); % get the number of marker
P = ones(len_p,4);
P(:,2:4) = original_p;
K = zeros(len_p,len_p);
for i = 1:len_p
    for j = 1:len_p
        r = sum(power(original_p(i,:)-original_p(j,:),2));
        K(i,j) = sqrt(max(r,1E-320));
    end
end

B = zeros(len_p+4,3);
B(1:len_p,:) = target_p;

A = [[K,P];[P',zeros(4,4)]];

W = pinv(A)*B;

len_v = size(vertices,1);
target_vertices = zeros(len_v,6);

for i = 1:len_v
    u = zeros(1,len_p+4);
    for j = 1:len_p
        r = sum(power(vertices(i,1:3)-target_p(j,:),2));
        u(1,j) = sqrt(max(r,1E-320));
    end
    u(1,len_p+1:len_p+4) = [1,vertices(i,1:3)];
    target_vertices(i,1:3) = u*W;
    target_vertices(i,4:6) = vertices(i,4:6);
end
min_cor_ori = min(vertices);
max_cor_ori = max(vertices);
len_cor_ori = max_cor_ori-min_cor_ori;
min_cor_tar = min(target_vertices);
max_cor_tar = max(target_vertices);
len_cor_tar = max_cor_tar-min_cor_tar;
scale = len_cor_ori(1:3)./len_cor_tar(1:3);
for i = 1:size(target_vertices,1)
    target_vertices(i,1:3) = (target_vertices(i,1:3)-min_cor_tar(1:3)).*scale+min_cor_ori(1:3);
end

save target_vertices target_vertices
end
        
    
