
function [outputmat]=cell2num2(inputcell)
% Function to convert an all numeric cell array to a double precision array
% ********************************************
% Usage: outputmatrix=cell2num(inputcellarray)
% ********************************************
% Output matrix will have the same dimensions as the input cell array
% Non-numeric cell contest will become NaN outputs in outputmat
% This function only works for 1-2 dimensional cell arrays
if ~iscell(inputcell), error('Input cell array is not.'); end
outputmat=zeros(size(inputcell));
for c=1:size(inputcell,2)
  for r=1:size(inputcell,1)
      if isnan(str2num(cell2mat(inputcell(r,c))))
%tf=isnan(A)：返回一个与A相同维数的数组，若A的元素为NaN（非数值），在对应位置上返回逻辑1（真），否则返回逻辑0（假）。
          outputmat(r,c)=0;
      else
          outputmat(r,c)=str2num(cell2mat(inputcell(r,c)));
      end
  end  
end
end