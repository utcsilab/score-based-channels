%TMULT Tensor matrix multiply
%
%   C = tmult(A, B, [transpose])
%
% Matrix multiplication over tensor arrays (i.e. arrays of matrices), with
% the ability to transpose matrices first.
%
%   C = tmult(A, B) is equivalent to:
%
%   sz = [size(B) 1];
%   sz(1) = size(A, 1);
%   C = zeros(sz);
%   for a = 1:prod(sz(3:end))
%       C(:,:,a) = A(:,:,a) * B(:,:,a);
%   end
%
% but is completely vectorized, so much faster. Tmult also supports
% bsxfun-style expansion of singular dimensions where appropriate, such
% that tmult(rand(4, 3, 10), rand(3, 2)) yields a 4x2x10 output.
%
%IN:
%   A - PxQx... array of input PxQ matrices, or QxP if transposed.
%   B - QxRx... array of input QxR matrices, or RxQ if transposed.
%   transpose - 2x1 logical array indicating which of the input matrices
%               need to be transposed before the multiply. Default: [0 0].
%
%OUT:
%   C - PxRx... output array.
function A = tmult(A, B, transpose)
szB = [size(B) 1];
szA = [size(A) 1];
if nargin < 3
    transpose = 0;
else
    transpose = [transpose(:); 0];
    transpose = [1 2] * (transpose(1:2) ~= 0);
    if transpose == 3
        % Permutation required. Choose the matrix which permutes fastest.
        pa = any(szA(1:2) == 1);
        pb = any(szB(1:2) == 1);
        if pa || (~pb && numel(A) < numel(B))
            if ~pa
                p = 1:numel(szA);
                p(1:2) = p([2 1]);
                A = permute(A, p);
            end
            szA(1:2) = szA([2 1]);
            transpose = 2;
        else
            if ~pb
                p = 1:numel(szB);
                p(1:2) = p([2 1]);
                B = permute(B, p);
            end
            szB(1:2) = szB([2 1]);
            transpose = 1;
        end
    end
end
switch transpose
    case 0
        % No transposes
        A = reshape(A, szA([1:2 end 3:end-1]));
        B = reshape(B, szB([end 1:end-1]));
        dim = 2;
        szB(1) = szA(1);
    case 1
        % First matrix transposed
        A = reshape(A, szA([1:2 end 3:end-1]));
        B = reshape(B, szB([1 end 2:end]));
        dim = 1;
        szB(1) = szA(2);
    case 2
        % Second matrix transposed
        A = reshape(A, szA([1 end 2:end]));
        B = reshape(B, szB([end 1:end-1]));
        dim = 3;
        szB(2) = szB(1);
        szB(1) = szA(1);
end
% Compute the output
A = sum(bsxfun(@times, A, B), dim);
% Reshape to expected size
szA = [szA ones(1, numel(szB)-numel(szA))];
szB = [szB ones(1, numel(szA)-numel(szB))];
szB(3:end) = max(szB(3:end), szA(3:end));
A = reshape(A, szB);
end