function tensor = vec_combination(V)
% Sum up the bias vectors to a 3-dimension tensor

tensor = V{1} + V{2}' + reshape(V{3},1,1,[]);
end