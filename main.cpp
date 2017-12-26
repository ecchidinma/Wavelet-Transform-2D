/* 
 * File:   main.cpp
 * Author: Emmanuel C. Chidinma
 * emmanuel.c.chidinma@gmail.com
 * 
 * 2D (Image) Haar discrete wavelet transform (DWT) and then the 2D inverse DWT
 *
 * This is optimized for DSP Processors and has been ported to an embedded
 * DSP platform; thus, in order to manage memory efficiently, NO scratch array is 
 * used: the transforms are done in-place. Although this is a C++ program, the use 
 * of classes, the bool variable type etc have intentionally been omitted
 * for easier porting to embedded (ANSI) C. For the same reason, recursive algorithms 
 * have been avoided. Furthermore, the use of OpenCV API, std::string class and 
 * std::stringstream class libraries is just to aid in the conversion to and from an 
 * image matrix to a *.jpg file for viewing and confirmation.  
 * While porting to embedded C, only the DWT and IDWT functions matter.
 *
 * length of the array(s) must be dyadic: a power of 2 eg, 2, 4, 8....1024 etc
 *
 * Created on December 4, 2016, 10:08 PM
 */
#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <cmath>
//#include <cstdlib>

//using namespace std;

const unsigned int NUM_ROWS = 256;
const unsigned int NUM_COLS = NUM_ROWS;
const unsigned long NUM_PIXELS = NUM_COLS * NUM_ROWS;
const float SQRT_2 = 1.414214f;
//const double SQRT_2 = 1.414213562373095;

unsigned char validateLength(unsigned short* pI, unsigned int length);
unsigned short inputAndValidation(unsigned short* pI);
void waveletTransform2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned short level);
void rearrange2DFromLR(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int decimatingColLen, unsigned int decimatingRowIndex);// decimatingRowIndex is zero-based row index
void revertRearrange2DFromLR(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int runningColLen, unsigned int runningRowIndex);// runningRowIndex is zero-based row index
void rearrange2DFromTC(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int decimatingRowLen, unsigned int decimatingColumnIndex);// decimatingColumnIndex is zero-based column index
void revertRearrange2DFromTC(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int runningRowLen, unsigned int runningColumnIndex); // runningColumnIndex is zero-based column index
void invWaveletTransform2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned short level);
unsigned int twoExpLevel(unsigned short iLevel);
void crtFlatArr(unsigned char* const pUch, const cv::Mat myImage);
void crtMatArr(unsigned char* const pUch, cv::Mat & myImage);
void printArr2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen);
/*
 * 
 */
int main() 
{
    // *.jpg test image filenames in project folder - USE ONE AT A TIME
    //std::string testImgFilenameJpg = "test_image";
    std::string testImgFilenameJpg = "test_image2";
    
    //read an RGB image
    cv::Mat imageRGB = cv::imread(testImgFilenameJpg + ".jpg");
    // convert it to a mono-channel (or monochrome) image
    cv::Mat image;
    cv::cvtColor(imageRGB, image, CV_BGR2GRAY);
    // create flattened array of the cv::Mat object
    unsigned char testArr2D[NUM_ROWS][NUM_COLS]; // alternatively use "new" to allocate dynamically
    unsigned char* pUch01 = testArr2D[0];
    crtFlatArr(pUch01, image);
    //print flattened input array image
    //printArr2D(testArr2D[0], NUM_ROWS, NUM_COLS);
    
    unsigned short iMaxLevel;
    unsigned short* pI01 = &iMaxLevel;
    
    if(validateLength(pI01, NUM_ROWS) <= 0) return 1;
    std::cout << "Maximum level " << " = " << iMaxLevel << std::endl << std::endl;
	
    //a) Ask the user to choose the DWT level
    //iMaxLevel = inputAndValidation(pI01);
    // OR 
    //b) Manually assign iMaxLevel if you do not want to input data in the command prompt
    iMaxLevel = 6;
   
    // Perform 2D Haar DWT
    waveletTransform2D(testArr2D[0], NUM_ROWS, NUM_COLS, iMaxLevel);
    //print DWT array image
    //printArr2D(testArr2D[0], NUM_ROWS, NUM_COLS);
    //create a Mat object with all pixels initially set to 255
    cv::Mat imageDWT = cv::Mat(NUM_ROWS, NUM_COLS, CV_8U, cv::Scalar(255));
    crtMatArr(pUch01, imageDWT); // populate imageDWT
    // create image window called "My DWT Image"
    cv::namedWindow("My DWT Image");
    // display resultant image on window
    cv::imshow("My DWT Image", imageDWT);
    // save resultant image
    cv::imwrite(testImgFilenameJpg + "_Level" + static_cast<std::ostringstream*>( &(std::ostringstream() << iMaxLevel) )->str() + "_DWT.jpg", imageDWT);
    
    // Perform 2D Haar IDWT
    invWaveletTransform2D(testArr2D[0], NUM_ROWS, NUM_COLS, iMaxLevel);
    //print IDWT image
    //printArr2D(testArr2D[0], NUM_ROWS, NUM_COLS);
    //create a Mat object with all pixels initially set to 255
    cv::Mat imageIDWT = cv::Mat(NUM_ROWS, NUM_COLS, CV_8U, cv::Scalar(255));
    crtMatArr(pUch01, imageIDWT); // populate imageIDWT
    // create image window called "My DWT Image"
    cv::namedWindow("My IDWT Image");
    // display resultant image on window
    cv::imshow("My IDWT Image", imageIDWT);
    // save resultant image
    cv::imwrite(testImgFilenameJpg + "_Level" + static_cast<std::ostringstream*>( &(std::ostringstream() << iMaxLevel) )->str() + "_IDWT.jpg", imageIDWT);
    
    //std::cin.get(); // press enter to close input screen (command prompt)
    
    // wait key for 0ms - that is wait indefinitely until any key is pressed to close OpenCV screen
    cv::waitKey(0);
    
    return 0;
}// end main())

void waveletTransform2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned short level)
{
    std::cout << "This is level " << level << " 2D DWT Computation." << std::endl << std::endl;
    unsigned int decimatingRowLen, decimatingColLen;
    
    decimatingColLen = colLen; // initial decimating column length deduced
    decimatingRowLen = rowLen; // initial decimating row length deduced
    
    while(level--)
    {
        //1) Perform 1D DWT row-wise, left to right
        for(unsigned int i = 0; i < decimatingRowLen; i++) //row indices for-loop
        {
            // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
            unsigned char iTemp01, iTemp02;
            for(unsigned int j = 0; j < decimatingColLen; j+=2) // only even column indices
            {
                int n = i*colLen + j; // nth index (flattened) in a 2D array in the direction from left to right row-wise
                int v = 1; // the index offset in a 2D array in the direction from left to right row-wise
                iTemp01 = *(arr + n); iTemp02 = *(arr + n + v);
                *(arr + n) = (int)(((iTemp01 + iTemp02)/SQRT_2) + 0.5); // calculate trend to the nearest int
                *(arr + n + v) = (int)(((iTemp01 - iTemp02)/SQRT_2) + 0.5); // calculate fluctuation to the nearest int
            }// //end column indices for-loop
            rearrange2DFromLR(arr, rowLen, colLen, decimatingColLen, i);
        }// //end row indices for-loop 

        //2) Then, perform 1D DWT column-wise, top to bottom
        for(unsigned int j = 0; j < decimatingColLen; j++) //column indices for-loop
        {
            // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
            unsigned char iTemp01, iTemp02;
            for(unsigned int i = 0; i < decimatingRowLen; i+=2) // only even row indices
            {
                int n = j + i*colLen; // nth index (flattened) in a 2D array in the direction from top to bottom column-wise
                int v = colLen; // the index offset in a 2D array in the direction from top to bottom column-wise
                iTemp01 = *(arr + n); iTemp02 = *(arr + n + v);
                *(arr + n) = (int)(((iTemp01 + iTemp02)/SQRT_2) + 0.5); // calculate trend to the nearest int
                *(arr + n + v) = (int)(((iTemp01 - iTemp02)/SQRT_2) + 0.5); // calculate fluctuation to the nearest int
            }// //end row indices for-loop
            rearrange2DFromTC(arr, rowLen, colLen,decimatingRowLen, j);
        }//end column indices for-loop 
        decimatingColLen /= 2; //OR: decimatingColLen >>= 1;
        decimatingRowLen /= 2; //OR: decimatingRowLen >>= 1;
    }// end while-loop
}// waveletTransform2D()

void invWaveletTransform2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned short level)
{
    std::cout << "This is level " << level << " IDWT Computation." << std::endl << std::endl;
    unsigned int runningRowLen, runningColLen; 
    
    runningRowLen = 2*rowLen/(1 << level); // initial row running length deduced
    //OR: runningRowLen = 2*rowLen/twoExpLevel(level); // initial row running length deduced
    runningColLen = 2*colLen/(1 << level); // initial column running length deduced
    //OR: runningColLen = 2*colLen/twoExpLevel(level); // initial column running length deduced		

    while(level--)
    {
        //1) Perform 1D IDWT column-wise, top to bottom
        for(unsigned int j = 0; j < runningColLen; j++) //column indices for-loop
        {
            // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
            revertRearrange2DFromTC(arr, rowLen, colLen, runningRowLen, j);
            unsigned char iTemp01, iTemp02;
            for(unsigned int i = 0; i < runningRowLen; i+=2) // only even row indices
            {
                    int n = j + i*colLen; // nth index (flattened) in a 2D array in the direction from top to bottom column-wise
                    int v = colLen; // the index offset in a 2D array in the direction from top to bottom column-wise
                    iTemp01 = *(arr + n); iTemp02 = *(arr + n + v);
                    *(arr + n) = (int)(((iTemp01 + iTemp02)/SQRT_2) + 0.5); // calculate sample to the nearest int
                    *(arr + n + v) = (int)(((iTemp01 - iTemp02)/SQRT_2) + 0.5); // calculate next sample to the nearest int
            } //end row indices for-loop  
        } //end column indices for-loop  

        //2) Then, perform 1D IDWT row-wise, left to right
        for(unsigned int i = 0; i < runningRowLen; i++) //row indices for-loop
        {
            // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
            revertRearrange2DFromLR(arr, rowLen, colLen, runningColLen, i);
            unsigned char iTemp01, iTemp02;
            for(unsigned int j = 0; j < runningColLen; j+=2) // only even column indices
            {
                    int n = i*colLen + j; // nth index (flattened) in a 2D array in the direction from left to right row-wise
                    int v = 1; // the index offset in a 2D array in the direction from left to right row-wise
                    iTemp01 = *(arr + n); iTemp02 = *(arr + n + v);
                    *(arr + n) = (int)(((iTemp01 + iTemp02)/SQRT_2) + 0.5); // calculate sample to the nearest int
                    *(arr + n + v) = (int)(((iTemp01 - iTemp02)/SQRT_2) + 0.5); // calculate next sample to the nearest int
            }// //end column indices for-loop
        }//end row indices for-loop
        runningRowLen *=2; // OR: runningRowLen <<= 1;
        runningColLen *=2; // OR: runningColLen <<= 1;
    }// end while-loop
}// end invWaveletTransform2D()

//A. re-arrange starting from left to right along row (row-wise)
void rearrange2DFromLR(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int decimatingColLen, unsigned int decimatingRowIndex)
{
    unsigned int i = decimatingRowIndex;
    //FOR AN ARRAY OF EVEN COLUMN LENGTH colLen (16), THE NUMBER OF ODD INDICES IS colLen/2 (8)
    // THEREFORE THE NUMBER OF ODD INDICES WITHIN THE LOWER-HALF-RANGE IS 4 (=colLen/4 ie 1, 3, 5, 7)
    // TO OBTAIN THEIR INDICES WITHIN A NEW ARRAY OF colLen/4 ELEMENTS, DO THE
    // FOLLOWING INTEGER MATH: 1/2, 3/2, 5/2, AND 7/2 WHICH YIELDS: 0, 1, 2, 3 
    int quartLen = decimatingColLen/4;
    unsigned char indexMask[quartLen]; // unsigned char used instead of bool to maintain compatibility with ANSI C
    // initialize to zero
    for(int k = 0; k < quartLen; k++)
    {
        indexMask[k] = 0;
    }// end for
   
    int lastMidOddIndex = (decimatingColLen/2) - 1;
    
    for(int j = 1; j <= lastMidOddIndex; j+=2)//consider only odd indices of the column up to the middle
    {//NB: j is the decimating column index, that is the index variable of the decimatingColLen 
        //IF THE CORRESPONDING FLAG TO THIS ODD INDEX IS NON-ZERO IT MEANS THAT INDEX HAS
        // ALREADY BEEN CONSIDERED. IN THAT CASE PLEASE MOVE ON TO THE NEXT ODD INDEX
        // OF COURSE ARRAY indexMask MUST NOT HAVE ZERO ALLOCATION (quartLen > 0)
        if((quartLen > 0) && indexMask[j/2]) continue; 

        int indexTemp01;
        float valueTemp;
        int indexTemp02 = j;

        //DEDUCE NEW INDEX LOCATIONS AND COPY ARRAY VALUES UNTIL WE RETURN TO THE STARTING INDEX
        do
        {
            if((indexTemp02 % 2) == 0) // true for even index
            {//if_else_a_begins.
                //calculate the new index for an old even index
                indexTemp01 = indexTemp02/2;

                if(indexTemp01 == j) //true if calculated index has matched initial index
                {// end if_else_a_begins
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = valueTemp;
                    break; //leave do-while loop
                }// end if_else_b_ctd.
                else
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = arr[(i*colLen)+indexTemp01]; //save the value whose new index location we shall find next to arr[(i*colLen)+j], now a scratch array space, as the temp location
                    arr[(i*colLen)+indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[(i*colLen)+j]; // save the value whose new index location we shall find next to the temp variable
                    indexTemp02 = indexTemp01; // assign index whose new index we shall find next
                }// if_else_b_ends
            }// if_else_a_ctd.
            else //((indexTemp02 % 2) != 0) // true for odd index
            {//NB: the 1st iteration of indexTemp02 (=j) must come here since j is always an odd index
                //calculate the new index for an old odd index
                indexTemp01 = indexTemp02 + lastMidOddIndex - (indexTemp02 - 1)/2;

                if(indexTemp02 == j) //this condition can only be true only once in this else clause of if_else_a block
                {// if_else_c_begins
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    valueTemp = arr[(i*colLen)+indexTemp01]; // save the value whose new index location we shall find next to the temp variable
                    // assign value at old index to that at the new index
                    arr[(i*colLen)+indexTemp01] = arr[(i*colLen)+indexTemp02]; // the implication of this is that arr[(i*colLen)+j] can now be used as a scratch space
                }// if_else_c_ctd.
                else // for subsequent values of indexTemp02 before next i from the outer for loop
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = arr[(i*colLen)+indexTemp01]; //save the value whose new index location we shall find next to arr[(i*colLen)+j], now a scratch array space, as the temp location
                    arr[(i*colLen)+indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[(i*colLen)+j]; // save the value whose new index location we shall find next to the temp variable

                    //CHECK WHETHER ANY POTENTIAL FUTURE ODD INDEX (THAT IS IN THE LOWER-HALF-RANGE) HAS BEEN DEDUCED FOR CONSIDERATION
                    // AND SET ITS FLAG SO THAT IT WILL NOT BE CONSIDERED AGAIN BY THE OUTER FOR LOOP
                    if((indexTemp02 <= lastMidOddIndex) && (quartLen > 0)) // odd index within mid range provided array indexMask doesn't have a zero allocation
                    {
                        indexMask[indexTemp02/2] = 1; // remember the integer math with odd number explained in the 1st few lines at the start of rearrange()
                    }//end if
                }// if_else_c_ends
                indexTemp02 = indexTemp01; // assign index whose new index we shall find next
            }// if_else_a_ends
        }while(indexTemp02 != j); // end do-while
    }// end column indices for-loop
}// end rearrange2DFromLR()

//AA. revert re-arrangement starting from left to right along row (row-wise)
void revertRearrange2DFromLR(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int runningColLen, unsigned int runningRowIndex)
{
    unsigned int i = runningRowIndex;
    //FOR AN ARRAY OF EVEN LENGTH colLen (16), THE NUMBER OF EVEN INDICES IS length/2 (ie 8,call it halfL)
    // THEREFORE THE NUMBER OF EVEN INDICES WITHIN THE UPPER-HALF-RANGE IS 4 (=length/4 ie 8, 10, 12, 14)
    // TO OBTAIN THEIR INDICES WITHIN A NEW ARRAY OF colLen/4 ELEMENTS, DO THE FOLLOWING INTEGER
    // MATH: (8-halfL)/2, (10-halfL)/2, (12-halfL)/2, AND (14-halfL)/2 WHICH YIELDS: 0, 1, 2, 3 
    int quartLen = runningColLen/4;
    unsigned char indexMask[quartLen];
    // initialize to zero
    for(int i = 0; i < quartLen; i++)
    {
        indexMask[i] = 0;
    }// end for
    
    int lastEvenIndex = runningColLen - 2;
    int halfLen = runningColLen/2; // this is also the first even index in the upper-half-range
    
    for(int j = halfLen; j <= lastEvenIndex; j+=2) //consider only even indices of the column from the middle up to the end
    {//NB: j is the running column index, that is the index variable of the runningColLen 
        //IF THE CORRESPONDING FLAG TO THIS EVEN INDEX IS NON-ZERO IT MEANS THAT INDEX HAS
        // ALREADY BEEN CONSIDERED. IN THAT CASE PLEASE MOVE ON TO THE NEXT ODD INDEX
        // OF COURSE ARRAY indexMask MUST NOT HAVE ZERO ALLOCATION (quartLen > 0)
        if((quartLen > 0) && indexMask[(j - halfLen)/2]) continue; 

        int indexTemp01;
        float valueTemp;
        int indexTemp02 = j;

        //DEDUCE NEW INDEX LOCATIONS AND COPY ARRAY VALUES UNTIL WE RETURN TO THE STARTING INDEX
        do
        {
            if(indexTemp02 < halfLen) // true for lower-half-range
            {//if_else_a_begins.
                //calculate the new index for an old lower-half-range index
                indexTemp01 = 2*indexTemp02;

                if(indexTemp01 == j) //true if calculated index has matched initial index
                {// end if_else_a_begins
                    //arr[j] = valueTemp;

                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = valueTemp;
                    break; //leave do-while loop
                }// end if_else_b_ctd.
                else
                {
                    // ASSIGN NEW POSITION
                    //arr[j] = arr[indexTemp01]; //save the value whose new index location we shall find next to arr[j], now a scratch array space, as the temp location
                    //arr[indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    //valueTemp = arr[j]; // save the value whose new index location we shall find next to the temp variable

                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = arr[(i*colLen)+indexTemp01]; //save the value whose new index location we shall find next to arr[(i*colLen)+j], now a scratch array space, as the temp location
                    arr[(i*colLen)+indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[(i*colLen)+j]; // save the value whose new index location we shall find next to the temp variable

                    indexTemp02 = indexTemp01; // assign index whose new index we shall find next
                }// if_else_b_ends
            }// if_else_a_ctd.
            else //(indexTemp02 >= halfLen) // true for upper-half-range
            {//NB: the 1st iteration of indexTemp02 (=j) must come here since j is always an upper-half-range index
                //calculate the new index for an old upper-half-range index
                indexTemp01 = 2*indexTemp02 - lastEvenIndex - 1;

                if(indexTemp02 == j) //this condition can only be true only once in this else clause of if_else_a block
                {// if_else_c_begins
                    // ASSIGN NEW POSITION
                    //valueTemp = arr[indexTemp01]; // save the value whose new index location we shall find next to the temp variable
                    // assign value at old index to that at the new index
                    //arr[indexTemp01] = arr[indexTemp02]; // the implication of this is that arr[j] can now be used as a scratch space

                     // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    valueTemp = arr[(i*colLen)+indexTemp01]; // save the value whose new index location we shall find next to the temp variable
                    // assign value at old index to that at the new index
                    arr[(i*colLen)+indexTemp01] = arr[(i*colLen)+indexTemp02]; // the implication of this is that arr[(i*colLen)+j] can now be used as a scratch space
                }// if_else_c_ctd.
                else // for subsequent values of indexTemp02 before next j from the outer for loop
                {
                    // ASSIGN NEW POSITION
                    //arr[j] = arr[indexTemp01]; //save the value whose new index location we shall find next to arr[j], now a scratch array space, as the temp location
                    //arr[indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    //valueTemp = arr[j]; // save the value whose new index location we shall find next to the temp variable

                     // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[(i*colLen)+j] = arr[(i*colLen)+indexTemp01]; //save the value whose new index location we shall find next to arr[(i*colLen)+j], now a scratch array space, as the temp location
                    arr[(i*colLen)+indexTemp01] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[(i*colLen)+j]; // save the value whose new index location we shall find next to the temp variable

                    //CHECK WHETHER ANY POTENTIAL FUTURE UPPER-HALF-RANGE INDEX (THAT IS EVEN) HAS BEEN DEDUCED FOR CONSIDERATION
                    // AND SET ITS FLAG SO THAT IT WILL NOT BE CONSIDERED AGAIN BY THE OUTER FOR LOOP
                    if(((indexTemp02 % 2) == 0) && (quartLen > 0)) // upper-half-range index that is even provided array indexMask doesn't have a zero allocation
                    {
                        indexMask[(indexTemp02 - halfLen)/2] = 1; // remember the integer math with odd number explained in the 1st few lines at the start of rearrange()
                    }//end if
                }// if_else_c_ends
                indexTemp02 = indexTemp01; // assign  index whose new index we shall find next
            }// if_else_a_ends
        }while(indexTemp02 != j); // end do-while
    }//end column indices for-loop
}// end revertRearrange2DFromLR()


//C. re-arrange starting from top to bottom along column (column-wise)
void rearrange2DFromTC(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int decimatingRowLen, unsigned int decimatingColumnIndex)
{
    unsigned int j = decimatingColumnIndex;
    //FOR AN ARRAY OF EVEN ROW LENGTH rowLen (16), THE NUMBER OF ODD INDICES IS rowLen/2 (8)
    // THEREFORE THE NUMBER OF ODD INDICES WITHIN THE LOWER-HALF-RANGE IS 4 (=rowLen/4 ie 1, 3, 5, 7)
    // TO OBTAIN THEIR INDICES WITHIN A NEW ARRAY OF length/4 ELEMENTS, DO THE
    // FOLLOWING INTEGER MATH: 1/2, 3/2, 5/2, AND 7/2 WHICH YIELDS: 0, 1, 2, 3 
    int quartLen = decimatingRowLen/4;
    unsigned char indexMask[quartLen]; // unsigned char used instead of bool to maintan compatibility with ANSI C
    // initialize to zero
    for(int k = 0; k < quartLen; k++)
    {
        indexMask[k] = 0;
    }// end for
   
    int lastMidOddIndex = (decimatingRowLen/2) - 1;
    
    for(int i = 1; i <= lastMidOddIndex; i+=2)//consider only odd row indices up to the middle
    {//NB: i is the decimating row index, that is the index variable of the decimatingRowLen 
        //IF THE CORRESPONDING FLAG TO THIS ODD INDEX IS NON-ZERO IT MEANS THAT INDEX HAS
        // ALREADY BEEN CONSIDERED. IN THAT CASE PLEASE MOVE ON TO THE NEXT ODD INDEX
        // OF COURSE ARRAY indexMask MUST NOT HAVE ZERO ALLOCATION (quartLen > 0)
        if((quartLen > 0) && indexMask[i/2]) continue; 

        int indexTemp01;
        float valueTemp;
        int indexTemp02 = i;

        //DEDUCE NEW INDEX LOCATIONS AND COPY ARRAY VALUES UNTIL WE RETURN TO THE STARTING INDEX
        do
        {
            if((indexTemp02 % 2) == 0) // true for even index
            {//if_else_a_begins.
                //calculate the new index for an old even index
                indexTemp01 = indexTemp02/2;

                if(indexTemp01 == i) //true if calculated index has matched initial index
                {// end if_else_a_begins
                    //arr[i] = valueTemp;

                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = valueTemp;
                    break; //leave do-while loop
                }// end if_else_b_ctd.
                else
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = arr[j+(indexTemp01*colLen)]; //save the value whose new index location we shall find next to arr[j+(i*colLen)], now a scratch array space, as the temp location
                    arr[j+(indexTemp01*colLen)] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[j+(i*colLen)]; // save the value whose new index location we shall find next to the temp variable

                    indexTemp02 = indexTemp01; // assign index whose new index we shall find next
                }// if_else_b_ends
            }// if_else_a_ctd.
            else //((indexTemp02 % 2) != 0) // true for odd index
            {//NB: the 1st iteration of indexTemp02 (=i) must come here since i is always an odd index
                //calculate the new index for an old odd index
                indexTemp01 = indexTemp02 + lastMidOddIndex - (indexTemp02 - 1)/2;

                if(indexTemp02 == i) //this condition can only be true only once in this else clause of if_else_a block
                {// if_else_c_begins
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    valueTemp = arr[j+(indexTemp01*colLen)]; // save the value whose new index location we shall find next to the temp variable
                    // assign value at old index to that at the new index
                    arr[j+(indexTemp01*colLen)] = arr[j+(indexTemp02*colLen)]; // the implication of this is that arr[j+(i*colLen)] can now be used as a scratch space
                }// if_else_c_ctd.
                else // for subsequent values of indexTemp02 before next i from the outer for loop
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = arr[j+(indexTemp01*colLen)]; //save the value whose new index location we shall find next to arr[j+(i*colLen)], now a scratch array space, as the temp location
                    arr[j+(indexTemp01*colLen)] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[j+(i*colLen)]; // save the value whose new index location we shall find next to the temp variable

                    //CHECK WHETHER ANY POTENTIAL FUTURE ODD INDEX (THAT IS IN THE LOWER-HALF-RANGE) HAS BEEN DEDUCED FOR CONSIDERATION
                    // AND SET ITS FLAG SO THAT IT WILL NOT BE CONSIDERED AGAIN BY THE OUTER FOR LOOP
                    if((indexTemp02 <= lastMidOddIndex) && (quartLen > 0)) // odd index within mid range provided array indexMask doesn't have a zero allocation
                    {
                        indexMask[indexTemp02/2] = 1; // remember the integer math with odd number explained in the 1st few lines at the start of rearrange()
                    }//end if
                }// if_else_c_ends
                indexTemp02 = indexTemp01; // assign  index whose new index we shall find next
            }// if_else_a_ends
        }while(indexTemp02 != i); // end do-while
    }// end row indices for-loop
}// rearrange2DFromTC()

//CC. revert re-arrangement starting from top to bottom along column (column-wise)
void revertRearrange2DFromTC(unsigned char* arr, unsigned int rowLen, unsigned int colLen, unsigned int runningRowLen, unsigned int runningColumnIndex)
{
    unsigned int j = runningColumnIndex;
    //FOR AN ARRAY OF EVEN ROW LENGTH rowLen (16), THE NUMBER OF EVEN INDICES IS length/2 (ie 8,call it halfL)
    // THEREFORE THE NUMBER OF EVEN INDICES WITHIN THE UPPER-HALF-RANGE IS 4 (=rowLen/4 ie 8, 10, 12, 14)
    // TO OBTAIN THEIR INDICES WITHIN A NEW ARRAY OF rowLen/4 ELEMENTS, DO THE FOLLOWING INTEGER
    // MATH: (8-halfL)/2, (10-halfL)/2, (12-halfL)/2, AND (14-halfL)/2 WHICH YIELDS: 0, 1, 2, 3 
    int quartLen = runningRowLen/4;
    unsigned char indexMask[quartLen];
    // initialize to zero
    for(int k = 0; k < quartLen; k++)
    {
        indexMask[k] = 0;
    }// end for
    
    int lastEvenIndex = runningRowLen - 2;
    int halfLen = runningRowLen/2; // this is also the first even index in the upper-half-range
    
    for(int i = halfLen; i <= lastEvenIndex; i+=2)//consider only even row indices from the middle up to the end
    {//NB: i is the running row index, that is the index variable of the runningRowLen 
        //IF THE CORRESPONDING FLAG TO THIS EVEN INDEX IS NON-ZERO IT MEANS THAT INDEX HAS
        // ALREADY BEEN CONSIDERED. IN THAT CASE PLEASE MOVE ON TO THE NEXT ODD INDEX
        // OF COURSE ARRAY indexMask MUST NOT HAVE ZERO ALLOCATION (quartLen > 0)
        if((quartLen > 0) && indexMask[(i - halfLen)/2]) continue; 

        int indexTemp01;
        float valueTemp;
        int indexTemp02 = i;

        //DEDUCE NEW INDEX LOCATIONS AND COPY ARRAY VALUES UNTIL WE RETURN TO THE STARTING INDEX
        do
        {
            if(indexTemp02 < halfLen) // true for lower-half-range
            {//if_else_a_begins.
                //calculate the new index for an old lower-half-range index
                indexTemp01 = 2*indexTemp02;

                if(indexTemp01 == i) //true if calculated index has matched initial index
                {// end if_else_a_begins
                    //arr[i] = valueTemp;

                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = valueTemp;
                    break; //leave do-while loop
                }// end if_else_b_ctd.
                else
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = arr[j+(indexTemp01*colLen)]; //save the value whose new index location we shall find next to arr[j+(i*colLen)], now a scratch array space, as the temp location
                    arr[j+(indexTemp01*colLen)] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[j+(i*colLen)]; // save the value whose new index location we shall find next to the temp variable

                    indexTemp02 = indexTemp01; // assign index whose new index we shall find next
                }// if_else_b_ends
            }// if_else_a_ctd.
            else //(indexTemp02 >= halfLen) // true for upper-half-range
            {//NB: the 1st iteration of indexTemp02 (=i) must come here since i is always an upper-half-range index
                //calculate the new index for an old upper-half-range index
                indexTemp01 = 2*indexTemp02 - lastEvenIndex - 1;

                if(indexTemp02 == i) //this condition can only be true only once in this else clause of if_else_a block
                {// if_else_c_begins
                    // ASSIGN NEW POSITION
                     // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    valueTemp = arr[j+(indexTemp01*colLen)]; // save the value whose new index location we shall find next to the temp variable
                    // assign value at old index to that at the new index
                    arr[j+(indexTemp01*colLen)] = arr[j+(indexTemp02*colLen)]; // the implication of this is that arr[j+(i*colLen)] can now be used as a scratch space
                }// if_else_c_ctd.
                else // for subsequent values of indexTemp02 before next i from the outer for loop
                {
                    // ASSIGN NEW POSITION
                    // INSTEAD USE MAPPED EQUIVALENT INDICES ONLY WHILE ASSIGNING VALUES
                    arr[j+(i*colLen)] = arr[j+(indexTemp01*colLen)]; //save the value whose new index location we shall find next to arr[j+(i*colLen)], now a scratch array space, as the temp location
                    arr[j+(indexTemp01*colLen)] = valueTemp; // assign the last saved value whose index was used to deduce indexTemp01
                    valueTemp = arr[j+(i*colLen)]; // save the value whose new index location we shall find next to the temp variable

                    //CHECK WHETHER ANY POTENTIAL FUTURE UPPER-HALF-RANGE INDEX (THAT IS EVEN) HAS BEEN DEDUCED FOR CONSIDERATION
                    // AND SET ITS FLAG SO THAT IT WILL NOT BE CONSIDERED AGAIN BY THE OUTER FOR LOOP
                    if(((indexTemp02 % 2) == 0) && (quartLen > 0)) // upper-half-range index that is even provided array indexMask doesn't have a zero allocation
                    {
                        indexMask[(indexTemp02 - halfLen)/2] = 1; // remember the integer math with odd number explained in the 1st few lines at the start of rearrange()
                    }//end if
                }// if_else_c_ends
                indexTemp02 = indexTemp01; // assign  index whose new index we shall find next
            }// if_else_a_ends
        }while(indexTemp02 != i); // end do-while
    }// end row indices for-loop
}//end revertRearrange2DFromTC()

unsigned char validateLength(unsigned short* pI, unsigned int length) // unsigned char used instead of bool to maintan compatibility with ANSI C
{
    char ch = (((length == 0) || (length == 1)) ? 'a' : (((length % 2) == 1) ? 'b' : 'c'));
    
    switch(ch)
    {
        case 'a':   
            std::cout << "Length of array cannot be 0 or 1" << std::endl;
            return 0;
        case 'b':
            std::cout << "Length of array cannot be odd" << std::endl;
            return 0;
        case 'c': // here, derive the log of length in base 2
            *pI = 1; // initialize contents of pointer to 1 since length is already even
            length /= 2;
            do
            {
                if((length % 2) == 1)
                {
                    std::cout << "Length of array is not a power of 2" << std::endl;
                    return 0;
                }
                (*pI)++;
                length /= 2;
            }while(length != 1); // end do-while
            return 1;
    }// end switch
}// end validateLength()

unsigned short inputAndValidation(unsigned short* pI)
{
   // assign an arbitrary, out-of-range value to ensure 
   // iLevel to ensure validation 
   unsigned short iLevel = *pI + 10;

   //validate the user's input
   while (!((0 <= iLevel)&&(iLevel <= *pI)))
   {
     iLevel = *pI; // default level value in case user did not enter any level number
     std::cout << "Please Choose a +ve DWT Level Less Than Or Equal To " << *pI << std::endl;
     std::cin >> iLevel;
     while (std::cin.get() != '\n');  // flush out any remaining newline xcters in input buffer
   }// end while
   return iLevel;
}//end inputAndValidation()

unsigned int twoExpLevel(unsigned short iLevel)
{
    unsigned int expValue = 1;
    for(int i=0; i<iLevel; i++)
    {
        expValue *= 2;
    }
    return expValue;
}//twoExpLevel()

// create flattened array
void crtFlatArr(unsigned char* const pUch, const cv::Mat myImage)
{
    int nr = myImage.rows;
    int nc = myImage.cols;
    
    std::cout << "Number of Channels = " << myImage.channels() << std::endl;
    
    for(int i = 0; i < nr; i++)
    {
        for(int j = 0; j < nc; j++)
        {
            *(pUch + i*nc + j) = myImage.at<uchar>(i, j);
        }// end column indices for-loop
    }// end row indices for-loop
}// end crtFlatArr()

// create cv::Mat array
void crtMatArr(unsigned char* const pUch, cv::Mat & myImage)
{
    int nr = myImage.rows;
    int nc = myImage.cols;
    
    //std::cout << "Number of Channels = " << myImage.channels() << std::endl;
    
    for(int i = 0; i < nr; i++)
    {
        for(int j = 0; j < nc; j++)
        {
            myImage.at<uchar>(i, j) = *(pUch + i*nc + j);
        }// end column indices for-loop
    }// end row indices for-loop
}// end crtMatArr()

void printArr2D(unsigned char* arr, unsigned int rowLen, unsigned int colLen)
{
    std::cout.right; // justify right
    int iWdt = 3; // width
    
    //std::cout << "X" << " "; // top-left mark
    std::cout.width(iWdt);
    std::cout << "X"; // top-left mark
    for(int colNumHeader = 0; colNumHeader < colLen; colNumHeader++)
    {
        //std::cout << colNumHeader << " ";
        std::cout.width(iWdt);
        std::cout << colNumHeader;
    }
    std::cout << std::endl << std::endl;

    for(int i = 0; i < rowLen; i++)
    {
        //std::cout << i << " "; // row number
        std::cout.width(iWdt); 
        std::cout << i;
        for(int j = 0; j < colLen; j++)
        {
            //std::cout << *(arr + i*colLen + j) << " "; // print values at position (i, j)
            std::cout.width(iWdt); 
            std::cout << static_cast<int>(*(arr + i*colLen + j)); // print values at position (i, j)
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}// end printArr()
