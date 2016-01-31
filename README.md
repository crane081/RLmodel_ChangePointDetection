# RLmodel_ChangePointDetection

This code is to generate stimuli sequence for three conditions in a change point detection task that is based on Yu & Huang 2014: http://www.cranehuang.com/uploads/5/0/5/9/50591795/yuhuang2014.pdf

1) no change point ('stable' environment)

2) change points occur based on N(30,1) (moderatly volatile environment)

3) change points occur based on N(10,1) (highly volatile environment)

Then use a standard TD model to simulate performance with a range of learning rate.
