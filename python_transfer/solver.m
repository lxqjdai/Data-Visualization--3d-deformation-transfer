function vsp1 = solver()
fprintf('Solving linear system in matlab...\n');
load './IandC.mat'
i11 = I1(:,1);
i12 =I1(:,2);
i13 = I1(:,3);
i21 = I2(:,1);
i22 =I2(:,2);
i23 = I2(:,3);

M1 = sparse(i11, i12, i13, max(i11), max(i12));
M2 = sparse(i21, i22, i23, max(i21), max(i22));
M3 = M1+M2;
clear i11 i12.i13 i21 i22 i23;
i31 = I3(:,1);
i32 =I3(:,2);
i33 = I3(:,3);
M4 = sparse(i31, i32, i33);

M = [M3 ; M4];
vsp1 = M\C';
fprintf('Solved!\n');
end