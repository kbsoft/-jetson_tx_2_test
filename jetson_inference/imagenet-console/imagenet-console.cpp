/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


std::vector<std::string> get_file_list() {
   std::ifstream myfile("output.txt");
   std::string line;
   std::vector<std::string> myLines;
   while (std::getline(myfile, line))
   {
      myLines.push_back(line);
   }
   return myLines;
}

// main entry point
int main( int argc, char** argv )
{
//	printf("imagenet-console\n  args (%i):  ", argc);
	
//	for( int i=0; i < argc; i++ )
//		printf("%i [%s]  ", i, argv[i]);
		
//	printf("\n\n");
	
	
	// retrieve filename argument
//	if( argc < 2 )
//	{
//		printf("imagenet-console:   input image filename required\n");
//		return 0;
//	}
	
//	const char* imgFilename = argv[1];
	
	// create imageNet
    imageNet* net = imageNet::Create(imageNet::VGG16);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}
	
	net->EnableProfiler();
    std::vector<std::string> v = get_file_list();
    std::ofstream logfile;
    logfile.open ("predictions.txt");
    for(int i = 0; i < v.size(); i++) {
        const char* imgFilename = v[i].c_str();
        // load image from file on disk
        float* imgCPU    = NULL;
        float* imgCUDA   = NULL;
        int    imgWidth  = 0;
        int    imgHeight = 0;

        if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
        {
            printf("failed to load image '%s'\n", imgFilename);
            return 0;
        }

        float confidence = 0.0f;
        // std::get<0>(mytuple)
        // classify image
        std::vector <std::tuple<int, float, double> > a = net->ClassifyFive(imgCUDA, imgWidth, imgHeight, &confidence);
        for (int j = 0; j < 5; j++) {
            logfile << imgFilename << "," << std::get<0>(a[j]) << "," << std::get<1>(a[j]) << "," << std::get<2>(a[j]) << "\n";
            //std::cout << imgFilename << "," << std::get<0>(a[j]) << "," << std::get<1>(a[j]) << "," << std::get<2>(a[j]) << "\n";
        }
        printf("\nshutting down...\n");
        CUDA(cudaFreeHost(imgCPU));
    }
    logfile.close();

	delete net;
	return 0;
}
