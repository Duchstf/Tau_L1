//Numpy array shape [80, 25]
//Min -1.515625000000
//Max 1.312500000000
//Number of zeros 1205

#ifndef W2_H_
#define W2_H_

#ifndef __SYNTHESIS__
weight2_t w2[2000];
#else
weight2_t w2[2000] = {0.000000, 0.000000, 0.000000, 0.109375, 0.046875, 0.078125, 0.000000, 0.281250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.046875, -0.125000, 0.000000, 0.703125, 0.031250, 0.000000, 0.000000, -0.203125, 0.000000, -1.296875, -0.656250, 0.000000, 0.000000, 0.390625, 0.000000, 0.000000, 0.203125, 0.125000, 0.000000, 0.000000, 0.000000, 0.234375, -0.343750, 0.000000, 0.203125, 0.000000, 0.359375, -1.515625, 1.312500, 0.546875, 0.000000, -0.046875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.156250, -0.125000, -0.250000, 0.000000, 0.187500, 0.000000, -0.906250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.718750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.578125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.093750, 0.000000, 0.390625, 0.406250, 0.625000, 0.000000, 0.000000, -0.140625, 0.000000, -0.125000, 0.000000, 0.000000, 0.000000, -0.546875, -0.109375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.218750, 0.000000, 0.000000, 0.000000, 0.390625, -0.265625, -0.234375, 0.000000, -0.265625, 0.000000, 0.000000, 0.562500, 0.000000, 0.140625, 0.000000, 0.000000, 0.296875, 0.000000, -0.812500, 0.000000, -0.375000, 0.000000, 0.000000, 0.062500, 0.234375, 0.000000, 0.000000, 0.062500, 0.000000, 0.000000, 0.062500, -0.140625, -0.078125, 0.187500, -0.906250, 0.000000, 0.000000, 0.000000, 0.000000, -0.281250, 0.000000, 0.718750, 0.296875, -0.937500, -0.937500, 0.000000, 0.000000, 0.000000, -0.062500, 0.843750, 0.031250, 0.000000, 0.468750, 0.000000, 0.000000, -0.484375, 0.000000, -0.656250, 0.875000, 0.000000, -0.109375, -0.015625, 0.000000, 0.000000, 0.140625, -0.343750, -0.421875, 0.343750, 0.078125, 0.000000, 0.000000, 0.171875, 0.000000, 0.000000, -0.531250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.375000, 0.000000, 0.000000, 0.000000, 0.218750, 0.546875, 0.015625, -0.109375, 0.000000, 0.000000, -0.125000, 0.000000, 0.000000, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.046875, 0.578125, 0.000000, 0.000000, 0.000000, -0.093750, 0.000000, 0.000000, 0.000000, 0.000000, 0.093750, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.312500, 0.000000, 0.000000, 0.000000, 0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.031250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.171875, 0.000000, -0.031250, 0.000000, 0.453125, 0.000000, 0.000000, 0.000000, 0.703125, 0.125000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.265625, 0.000000, 0.000000, -0.484375, 0.000000, -0.031250, 0.000000, 0.000000, -0.156250, 0.000000, 0.187500, 0.484375, 0.343750, 0.000000, 0.000000, 0.000000, 0.000000, 0.015625, 0.000000, 0.000000, 0.156250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.140625, 0.000000, 0.000000, -0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.390625, 0.000000, -0.125000, -0.156250, 0.000000, 0.000000, 0.000000, -0.062500, 0.000000, 0.218750, -0.671875, 0.000000, 0.281250, 0.000000, 0.000000, 0.000000, 0.125000, -0.125000, 0.000000, 0.000000, 0.000000, 0.000000, 0.203125, 0.593750, 0.000000, 0.000000, 0.000000, 0.000000, 0.281250, 0.000000, -0.140625, -0.171875, -0.265625, 0.000000, 0.437500, -0.343750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.140625, 0.000000, 0.000000, 0.000000, -0.281250, -0.125000, 0.000000, 1.234375, -0.171875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.234375, 0.078125, 0.000000, -0.546875, 0.421875, 0.000000, -0.312500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.250000, -0.375000, 0.000000, 0.000000, 0.515625, 0.000000, 0.000000, 0.000000, -0.171875, 0.000000, -0.515625, -0.156250, 0.000000, 0.171875, -0.453125, 0.000000, 0.000000, -0.500000, 0.000000, 0.171875, -0.187500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.296875, 0.187500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.156250, 0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.156250, 0.187500, 0.000000, 0.000000, 0.000000, -0.421875, 0.000000, -0.281250, 0.000000, 0.000000, 0.000000, -0.187500, 0.000000, 0.000000, 0.187500, 0.000000, 0.000000, 0.359375, 0.000000, 0.000000, 0.187500, -0.093750, 0.000000, 0.000000, 0.046875, -0.250000, 0.000000, 0.000000, 0.156250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.031250, 0.000000, 0.000000, -0.250000, 0.000000, 0.000000, -0.125000, 0.000000, 0.000000, 0.000000, 0.000000, -0.078125, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.015625, 0.000000, 0.000000, -0.406250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.109375, 0.000000, 0.000000, -0.218750, -0.359375, 0.000000, 0.421875, 0.000000, 0.062500, 0.000000, -0.421875, -0.046875, 0.000000, 0.093750, 0.000000, 0.000000, 0.000000, -0.515625, 0.000000, 0.000000, 0.015625, 0.000000, 0.000000, -0.218750, 0.281250, 0.000000, 0.281250, -0.156250, 0.250000, 0.000000, 0.000000, 0.109375, 0.015625, 0.000000, 0.000000, 0.187500, 0.000000, 0.406250, -0.062500, -0.281250, -0.078125, 0.000000, 0.000000, 0.000000, -0.250000, -0.453125, -0.046875, 0.421875, 0.000000, -0.109375, 0.109375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.281250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.218750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.656250, 0.000000, 0.000000, 0.000000, 0.046875, 0.000000, 0.484375, -0.546875, -0.031250, -0.421875, 0.000000, -0.781250, 0.000000, 0.000000, -0.546875, 0.265625, 0.171875, -0.203125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.171875, 0.125000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.328125, 0.000000, 0.140625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.062500, 0.171875, 0.000000, 0.015625, 0.000000, 0.000000, 0.000000, 0.000000, -0.234375, 0.000000, 0.000000, 0.000000, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.093750, 0.203125, -0.140625, 0.000000, -0.328125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.359375, 0.000000, 0.187500, -0.171875, -0.187500, 0.000000, 0.031250, 0.000000, 0.000000, 0.125000, 0.015625, 0.000000, 0.000000, 0.000000, 0.000000, -0.109375, 0.000000, -0.296875, 0.000000, -0.187500, 0.250000, 0.000000, 0.000000, 0.390625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.078125, 0.000000, 0.000000, -0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.093750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.250000, 0.000000, -0.109375, -0.296875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.234375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.187500, 0.000000, 0.125000, 0.000000, 0.109375, -0.750000, 0.125000, 0.000000, -0.187500, 0.156250, 0.000000, 0.109375, 0.000000, 0.109375, 0.000000, -0.265625, -0.031250, 0.000000, 0.125000, 0.000000, 0.203125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.187500, 0.000000, -0.203125, 0.000000, 0.000000, 0.187500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.062500, 0.031250, 0.015625, 0.000000, 0.078125, -0.328125, -0.031250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.015625, 0.000000, -0.468750, -0.031250, 0.000000, 0.000000, -0.046875, 0.703125, 0.000000, -0.093750, 0.265625, 0.312500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.328125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.375000, 0.000000, -0.187500, 0.109375, 0.000000, 0.000000, 0.125000, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, -0.250000, -0.156250, 0.000000, 0.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.187500, 0.000000, 0.109375, 0.000000, 0.031250, 0.031250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.234375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.171875, 0.000000, -0.328125, 0.000000, 0.000000, 0.000000, 0.000000, -0.281250, 0.000000, 0.000000, -0.265625, 0.000000, 0.125000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.312500, 0.000000, 0.000000, 0.000000, 0.000000, -0.031250, 0.000000, 0.390625, 0.000000, 0.000000, -0.343750, 0.000000, 0.000000, -0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.046875, -0.109375, 0.171875, -0.031250, 0.125000, 0.000000, 0.000000, -0.359375, -0.171875, -0.328125, 0.000000, 0.000000, 0.218750, 0.281250, -0.437500, 0.000000, 0.000000, 0.109375, -0.093750, 0.000000, -0.125000, 0.000000, 0.000000, 0.140625, 0.156250, 0.000000, 0.000000, 0.375000, 0.000000, 0.156250, 0.000000, 0.000000, 0.265625, 0.000000, 0.000000, 0.296875, 0.000000, 0.000000, -0.140625, 0.000000, -0.250000, 0.000000, -0.187500, 0.296875, 0.000000, -0.218750, 0.000000, 0.218750, 0.000000, -0.171875, -0.218750, 0.000000, -0.328125, 0.000000, 0.062500, 0.000000, 0.234375, 0.000000, 0.000000, 0.000000, -0.062500, 0.000000, 0.000000, 0.265625, 0.000000, 0.000000, 0.000000, 0.046875, 0.000000, 0.000000, 0.093750, -0.265625, 0.000000, 0.265625, -0.359375, 0.000000, 0.000000, 0.062500, 0.000000, 0.140625, 0.000000, 0.046875, 0.000000, 0.000000, 0.000000, 0.156250, 0.000000, 0.203125, 0.000000, 0.000000, -0.203125, 0.000000, 0.000000, 0.328125, 0.000000, -0.484375, 0.000000, 0.281250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.390625, 0.000000, 0.000000, 0.000000, 0.000000, -0.031250, 0.000000, -0.375000, -0.156250, 0.000000, -0.015625, 0.000000, -0.421875, 0.000000, 0.093750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.312500, 0.000000, 0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.390625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.125000, 0.046875, 0.000000, 0.000000, 0.000000, -0.296875, 0.000000, 0.000000, 0.000000, 0.078125, 0.000000, 0.000000, 0.281250, 0.156250, 0.000000, 0.000000, -0.156250, 0.218750, 0.375000, 0.000000, 0.000000, 0.000000, -0.015625, -0.125000, 0.015625, 0.359375, 0.171875, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.031250, 0.000000, 0.187500, 0.000000, 0.000000, 0.000000, 0.000000, 0.281250, 0.171875, 0.000000, 0.171875, 0.000000, -0.078125, -0.312500, 0.000000, 0.000000, 0.000000, 0.000000, -0.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.031250, 0.000000, 0.296875, 0.000000, 0.000000, -0.265625, 0.000000, 0.000000, 0.343750, 0.250000, 0.000000, -0.265625, 0.000000, 0.000000, 0.000000, 0.000000, -0.109375, 0.000000, 0.078125, 0.000000, -0.140625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.187500, 0.000000, 0.078125, -0.156250, 0.234375, -0.031250, 0.000000, 0.234375, 0.000000, 0.109375, 0.031250, 0.000000, -0.187500, 0.000000, 0.093750, 0.343750, -0.062500, 0.000000, -0.015625, 0.093750, 0.000000, 0.000000, 0.000000, 0.015625, 0.015625, 0.000000, 0.140625, 0.234375, 0.156250, 0.000000, 0.000000, -0.062500, 0.187500, 0.000000, 0.000000, 0.000000, 0.015625, -0.125000, 0.000000, 0.234375, -0.234375, -0.171875, 0.171875, 0.015625, 0.000000, 0.375000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.171875, -0.218750, 0.000000, 0.171875, 0.000000, 0.000000, -0.718750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.328125, 0.000000, 0.218750, -0.265625, 0.000000, 0.000000, 0.000000, 0.187500, 0.000000, 0.109375, 0.000000, -0.171875, 0.062500, 0.000000, 0.000000, 0.093750, 0.000000, 0.000000, -0.078125, 0.000000, -0.328125, 0.000000, 0.000000, 0.187500, 0.000000, -0.359375, -0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.203125, -0.093750, 0.000000, 0.078125, 0.000000, 0.000000, -0.187500, 0.000000, 0.000000, 0.000000, -0.109375, 0.000000, 0.312500, 0.187500, 0.000000, 0.000000, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.203125, -0.171875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.140625, 0.156250, 0.187500, 0.000000, 0.109375, 0.000000, 0.000000, 0.000000, 0.000000, 0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.140625, 0.000000, -0.156250, 0.000000, 0.312500, 0.000000, 0.000000, 0.000000, 0.000000, 0.031250, -0.031250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.046875, 0.062500, 0.000000, 0.000000, 0.000000, -0.015625, -0.125000, 0.000000, 0.000000, 0.000000, -0.156250, -0.015625, 0.250000, -0.109375, -0.171875, 0.000000, 0.000000, 0.000000, 0.000000, 0.343750, 0.437500, -0.031250, 0.093750, 0.000000, -0.250000, 0.031250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.078125, 0.031250, 0.000000, 0.000000, 0.437500, 0.000000, 0.140625, 0.296875, 0.125000, 0.000000, -0.078125, -0.156250, 0.000000, 0.000000, -0.109375, 0.000000, -0.156250, -0.062500, 0.203125, -0.062500, 0.000000, 0.140625, -0.125000, 0.218750, 0.000000, 0.000000, -0.421875, 0.000000, 0.000000, 0.312500, 0.000000, 0.000000, 0.234375, 0.250000, 0.000000, 0.000000, 0.000000, 0.406250, 0.000000, -0.062500, 0.000000, 0.015625, 0.000000, 0.000000, 0.000000, -0.281250, 0.000000, 0.515625, 0.000000, 0.000000, 0.484375, 0.187500, 0.000000, -0.218750, 0.000000, 0.312500, 0.000000, 0.125000, 0.062500, 0.125000, 0.000000, 0.468750, -0.578125, 0.000000, -0.546875, -0.265625, 0.000000, 0.000000, 0.000000, -0.328125, 0.234375, 0.296875, -0.468750, 0.000000, 0.000000, 0.187500, 0.000000, 0.000000, 0.046875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.093750, 0.000000, 0.640625, -0.421875, 0.000000, -0.296875, 0.000000, 0.000000, 0.093750, 0.000000, -0.234375, 0.000000, 0.000000, -0.281250, -0.265625, 0.000000, -0.250000, 0.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.046875, 0.015625, 0.000000, 0.000000, 0.156250, 0.000000, 0.000000, 0.250000, -0.062500, 0.000000, 0.000000, 0.000000, 0.187500, 0.000000, 0.000000, 0.000000, 0.250000, 0.000000, -0.140625, -0.125000, 0.000000, 0.250000, 0.000000, 0.000000, -0.343750, 0.000000, 0.343750, 0.000000, 0.000000, -0.140625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.281250, 0.000000, 0.000000, 0.234375, 0.000000, 0.078125, -0.515625, 0.000000, -0.234375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.375000, 0.000000, -0.046875, 0.000000, 0.296875, 0.296875, 0.000000, 0.109375, 0.312500, 0.000000, -0.281250, 0.000000, 0.109375, 0.218750, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.281250, 0.000000, -0.109375, 0.000000, 0.203125, 0.000000, -0.046875, 0.000000, 0.000000, 0.000000, -0.281250, 0.000000, -0.234375, 0.515625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.109375, 0.109375, 0.000000, -0.031250, -0.156250, -0.296875, 0.000000, -0.390625, 0.000000, 0.171875, -0.093750, 0.000000, 0.312500, 0.312500, 0.000000, -0.125000, 0.000000, 0.171875, -0.093750, 0.125000, 0.000000, 0.000000, 0.000000, 0.203125, 0.000000, 0.046875, 0.000000, -0.281250, -0.281250, -0.265625, 0.000000, 0.000000, 0.000000, 0.343750, 0.000000, 0.000000, 0.000000, -0.140625, 0.000000, 0.359375, 0.203125, 0.000000, -0.125000, 0.000000, 0.000000, -0.140625, -0.046875, 0.171875, 0.421875, -0.078125, 0.187500, 0.000000, 0.000000, 0.250000, 0.156250, 0.000000, -0.234375, -0.500000, 0.031250, 0.265625, 0.390625, -0.453125, 0.000000, 0.000000, 0.000000, -0.296875, -0.109375, -0.390625, 0.000000, -0.250000, 0.000000, 0.000000, -0.203125, 0.000000, 0.250000, -0.234375, 0.000000, -0.078125, 0.265625, 0.140625, -0.140625, 0.000000, 0.000000, 0.000000, 0.281250, 0.546875, 0.000000, 0.000000, 0.000000, 0.000000, -0.359375, 0.000000, -0.328125, 0.156250, -0.296875, 0.171875, 0.000000, 0.000000, 0.171875, 0.000000, -0.625000, 0.000000, 0.000000, 0.000000, 0.000000, 0.171875, -0.140625, 0.000000, -0.187500, 0.000000, 0.078125, -0.281250, 0.187500, 0.000000, 0.125000, -0.093750, 0.000000, 0.203125, -0.203125, 0.000000, -0.187500, 0.031250, 0.000000, -0.156250, -0.078125, -0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.281250, 0.000000, 0.078125, 0.000000, 0.296875, 0.000000, 0.000000, -0.234375, 0.000000, -0.015625, 0.000000, 0.078125, 0.281250, 0.000000, 0.171875, 0.109375, 0.000000, 0.203125, -0.406250, -0.187500, 0.000000, 0.000000, 0.000000, -0.328125, 0.046875, 0.000000, 0.296875, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.234375, 0.265625, -0.156250, -0.203125, 0.000000, -0.281250, 0.171875, 0.000000, 0.000000, 0.375000, -0.062500, 0.000000, 0.406250, 0.000000, 0.296875, -0.031250, -0.046875, 0.000000, 0.328125, 0.000000, 0.000000, 0.000000, 0.218750, -0.109375, 0.000000, 0.000000, 0.125000, -0.093750, -0.125000, 0.000000, -0.171875, 0.234375, -0.140625, 0.000000, 0.062500, -0.015625, 0.000000, 0.156250, -0.453125, 0.000000, 0.000000, 0.109375, -0.140625, 0.109375, 0.000000, 0.312500, 0.000000, -0.171875, 0.125000, -0.250000, 0.000000, 0.187500, -0.078125, 0.156250, 0.000000, 0.125000, 0.203125, 0.000000, -0.281250, 0.000000, 0.000000, 0.000000, 0.000000, 0.109375, 0.000000, 0.000000, 0.250000, 0.187500, 0.000000, 0.171875, -0.109375, 0.000000, 0.000000, -0.031250, 0.000000, 0.187500, 0.203125, 0.000000, 0.000000, 0.000000, 0.062500, 0.093750, 0.156250, 0.000000, 0.015625, 0.000000, 0.515625, 0.328125, 0.000000, -0.015625, 0.000000, 0.000000, 0.312500, 0.484375, 0.000000, 0.000000, -0.312500, 0.000000, -0.531250, -0.250000, -0.140625, 0.125000, 0.000000, 0.406250, 0.000000, 0.000000, 0.171875, 0.296875, -0.875000, 0.000000, -0.281250, 0.359375, 0.000000, 0.000000, 0.000000, -0.375000, 0.000000, 0.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.500000, 0.000000, -0.140625, 0.000000, -0.156250, 0.000000, 0.000000, -0.218750, 0.296875, 0.000000, -1.109375, 0.000000, 0.000000, 0.000000, 0.250000, 0.000000, -0.046875, 0.062500, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.328125, 0.000000, 0.109375, 0.000000, -0.109375, 0.531250, 0.203125, 0.000000, 0.000000, 0.156250, -0.203125, -0.484375, 0.000000, 0.000000, -0.281250, 0.000000, 0.000000, -0.156250, 0.000000, 0.000000, -0.078125, 0.203125, -0.109375, 0.000000, 0.000000, 0.000000, -0.062500, -0.062500, 0.000000, 0.000000, 0.000000, -0.109375, -0.250000, 0.000000, 0.000000, 0.000000, -0.046875, -0.281250, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.093750, 0.328125, 0.000000, 0.312500, 0.000000, 0.000000, 0.000000, 0.515625, -0.187500, 0.000000, 0.062500, -0.031250, -0.250000, 0.281250, -0.015625, -0.312500, 0.000000, 0.140625, 0.000000, 0.000000, 0.000000, 0.218750, 0.375000, -0.203125, 0.343750, 0.000000, 0.125000, 0.125000, 0.359375, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.093750, 0.312500, 0.000000, 0.000000, 0.250000, 0.000000, 0.359375, -0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.312500, 0.000000, 0.000000, -0.125000, 0.156250, 0.000000, -0.781250, 0.000000, 0.000000, -0.093750, 0.156250, -0.328125, 0.078125, 0.125000, 0.140625, 0.000000, 0.109375, -0.187500, 0.171875, 0.140625, -0.265625, -0.234375, 0.000000, 0.000000, 0.000000, 0.000000, -0.093750, 0.000000, 0.000000, 0.171875, 0.000000, 0.078125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.218750, -0.171875, 0.000000, -0.187500, -0.234375, 0.000000, 0.140625, 0.156250, 0.000000, 0.484375, 0.406250, -0.234375, 0.343750, -0.812500, 0.000000, 0.000000, 0.000000, -1.265625, 0.000000, 0.000000, 0.000000, -0.421875, 0.000000, 0.000000, 0.000000, -0.265625, 0.000000, 0.000000, 0.000000, 0.281250, 0.000000, 0.000000, -0.843750, 0.000000};
#endif

#endif
