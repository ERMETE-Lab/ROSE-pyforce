/* 
                        BENCHMARK PROBLEM
           
  Identification: 11-A2          Source Situation ID.11
  Date Submitted: June 1976      By: R. R. Lee (CE)
                                     D. A. Menely (Ontario Hydro)
                                     B. Micheelsen (Riso-Denmark)
                                     D. R. Vondy (ORNL)
                                     M. R. Wagner (KWU)
                                     W. Werner (GRS-Munich)
 
  Date Accepted:  June 1977      By: H. L. Dodds, Jr. (U. of Tenn.)
                                     M. V. Gregory (SRL)
 
  Descriptive Title: Two-dimensional LWR Problem,
                     also 2D IAEA Benchmark Problem
 
  Reduction of Source Situation
            1. Two-groupo diffusion theory
            2. Two-dimensional (x,y)-geometry
*/

lc = 1.5; // the characteristic length of the mesh (to be expanded by m4)

// geometry
Point(1) = {0, 0, 0, lc};
Point(2) = {10, 0, 0, lc};
Point(4) = {10, 10, 0, lc};
Point(5) = {70, 0, 0, lc};
Point(6) = {70, 10, 0, lc};
Point(7) = {90, 10, 0, lc};
Point(8) = {90, 0, 0, lc};
Point(13) = {130, 0, 0, lc};
Point(14) = {130, 30, 0, lc};
Point(15) = {110, 30, 0, lc};
Point(16) = {110, 70, 0, lc};
Point(17) = {90, 70, 0, lc};
Point(18) = {90, 90, 0, lc};
Point(20) = {70, 70, 0, lc};
Point(21) = {110, 110, 0, lc};
Point(31) = {110, 90, 0, lc};
Point(32) = {130, 90, 0, lc};
Point(33) = {130, 50, 0, lc};
Point(34) = {150, 50, 0, lc};
Point(35) = {150, 0, 0, lc};
Point(36) = {170, 0, 0, lc};
Point(37) = {170, 70, 0, lc};
Point(38) = {150, 70, 0, lc};
Point(39) = {150, 110, 0, lc};
Point(40) = {130, 110, 0, lc};
Point(41) = {130, 130, 0, lc};//+
Line(1) = {1, 2};
//+
Line(2) = {2, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {2, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 5};
//+
Line(9) = {8, 13};
//+
Line(10) = {13, 14};
//+
Line(11) = {14, 15};
//+
Line(12) = {15, 16};
//+
Line(13) = {16, 17};
//+
Line(14) = {17, 20};
//+
Line(15) = {20, 18};
//+
Line(16) = {18, 17};
//+
Line(17) = {13, 35};
//+
Line(18) = {35, 34};
//+
Line(19) = {34, 33};
//+
Line(20) = {36, 37};
//+
Line(21) = {35, 36};
//+
Line(22) = {37, 38};
//+
Line(23) = {38, 39};
//+
Line(24) = {39, 40};
//+
Line(25) = {40, 41};
//+
Line(26) = {33, 32};
//+
Line(27) = {32, 31};
//+
Line(28) = {31, 21};
//+
Line(29) = {18, 21};
//+
Line(30) = {21, 41};
//+
Line(31) = {20, 4};
//+
Curve Loop(1) = {31, -2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14};
//+
Curve Loop(2) = {16, -13, -12, -11, -10, 17, 18, 19, 26, 27, 28, -29};
//+
Curve Loop(3) = {2, 3, 1};
//+
Curve Loop(4) = {6, 7, 8, 5};
//+
Curve Loop(5) = {15, 16, 14};
//+
Curve Loop(6) = {30, -25, -24, -23, -22, -20, -21, 18, 19, 26, 27, 28};
//+
Plane Surface(1) = {1};
//+
Plane Surface(2) = {2};
//+
Plane Surface(3) = {3};
//+
Plane Surface(4) = {4};
//+
Plane Surface(5) = {5};
//+
Plane Surface(6) = {6};
//+
Physical Surface("fuel1", 1) = {1};
//+
Physical Surface("fuel2", 2) = {2};
//+
Physical Surface("fuel-rod", 3) = {3, 4, 5};
//+
Physical Surface("reflector", 4) = {6};
//+
Physical Curve("void", 10) = {24, 25, 23, 22, 20};
//+
Physical Curve("sym", 20) = {31, 3, 15, 29, 30, 1, 4, 8, 9, 17, 21};

Mesh.Light = 0;
General.SmallAxes = 0;

// meshing options
Mesh.Algorithm= 6; 
Mesh.Optimize = 1;

