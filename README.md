# Wavelet-Transform-2D

•	Project (Description): 2D (Image) Haar Discrete Wavelet Transform (DWT) and then the 2D Inverse DWT

•	Synopsis: 
Although this program can be run on the desktop PC, it is optimized for DSP Processors and has actually been ported to an embedded DSP platform; thus, in order to manage memory efficiently, NO scratch arrays were used: the transforms are done in-place. Although this is a C++ program, the use of classes, the bool variable type and advanced C++ features, have intentionally been omitted for easier porting to embedded (ANSI) C. For the same reason, recursive algorithms have been avoided. Furthermore, the use of OpenCV API, std::string class and  std::stringstream class libraries is just to aid in the conversion to and from an image matrix to a *.jpg file for viewing and confirmation. While porting to embedded C, only the DWT and IDWT functions are needed.

•	Programming Languages and IDEs: 
a)	C and C++.
b)	OpenCV API
c)	MATLAB
d)	Net Beans IDE

•	How to use:
The input image must be a square image with dyadic dimensions. If the image does not meet these criteria, it should be padded to the nearest power of two.

•	Feedback:
Please send me an email at: emmanuel.c.chidinma@gmail.com
