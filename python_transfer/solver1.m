function x = solver1()
fprintf('Solving second equation on matlab...\n');
load './IandC2.mat'
i1 = I(:,1);
i2 =I(:,2);
i3 = I(:,3);

M = sparse(i1, i2, i3, max(i1), max(i2));
x = (M'*M)\(M'*C);


fprintf('Solved!\n');
end