function [ TI,BC ] = locate_particles( TR, QP, TI )
%LOCATE_PARTICLES Locate Particles in Triangulation using last known cell
% Author: *Jakob Ameres* <http://jakobameres.com jakobameres.com>
%   _TR_ Triangulation
%   _QP_ query point
%   _TI_ Triangle index, last known position

%Get barycentric coordinates
BC=cartesianToBarycentric(TR,TI,QP);

% Get most negative coordinate
[BCMIN, BCIDX]=min(BC,[],2);

%
% points with negative barycentric coordinate 
% are lying outside of the triangle specified by TI
out=(BCMIN<0);


% If any particle lies outside
if any(out)

%Find the neighbors of the triangle
TN=neighbors(TR,TI(out));


% Set new triangle index to neighbour corresponding to the most
% negative barycentric coordinate
TI(out)= TN(sub2ind(size(TN), (1:size(TN,1))', BCIDX(out)));

[TI_up,BC_up]=locate_particles(TR,QP(out,:),TI(out));


TI(out)=TI_up;
 BC(out,:)=BC_up;
end

end

