#include <iostream>
#include <cassert>

using namespace std;

#include "Shape.hpp"

using namespace lmlib;

#define display(expr) do { cout << #expr"= " << expr << endl; } while (0);

void unittest_shape()
{
	Shape<3> a = Shape3(5, 3, 4);
	Shape<3> b = Shape3(5, 4, 3);
	Shape<5> c = Shape5(5, 6, 7, 4, 3);
	display(a);
	display(b);
	display(c);
	display((c.slice<2, 5>()));
	assert(a != b);
	assert(a.size() == b.size());
	assert(a.flat_to_1D() == b.flat_to_1D());
	assert(a.flat_to_2D() == Shape2(15, 4));
	assert(a.subshape() == Shape2(3, 4));
	assert((c.slice<2, 5>()) == Shape3(7, 4, 3));
	cout << "unittest_shape complete.\n";
}


int main()
{
	unittest_shape();
}