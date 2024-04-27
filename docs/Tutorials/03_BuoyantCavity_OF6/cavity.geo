Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {3, 4, 1, 2};
Plane Surface(1) = {1};

Physical Point("pressure_value", 5) = {1};
Physical Curve("lid", 10) = {3};
Physical Curve("walls", 20) = {4, 1, 2};

Physical Surface("domain", 100) = {1};

Transfinite Surface {1};
//Mesh.MeshSizeFactor = 0.01;
//Transfinite Curve {4, 2, 3, 1} = 100 Using Progression 1;
//+
Transfinite Curve {3, 1, 4, 2} = 128 Using Progression 1;
