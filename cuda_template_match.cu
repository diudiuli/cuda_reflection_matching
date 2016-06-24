//Author: Dongwei Shi
//Created: 06/15/2016
//Description: this program is for template matching with cuda. The program is expected to template match several template simutaneously

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include </usr/local/cuda-7.5/include/cuda.h>

#define KERNEL_WIDTH 25
#define KERNEL_RADIUS 12
#define TILE_WIDTH 6
#define BLK_SIZE (TILE_WIDTH+KERNEL_WIDTH-1)
#define IMGSIZE 307200
//#include 
using namespace std;
using namespace cv;
//global image and templates
Mat img, gray_img;
Mat templs[256];
__constant__ float deviceKernel[KERNEL_WIDTH*KERNEL_WIDTH];

__global__ void conv2d(float* A, float* B, const int x_size, const int y_size)
{
   
    __shared__ float Nds[BLK_SIZE][BLK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x_out = bx*TILE_WIDTH + tx;
    int y_out = by*TILE_WIDTH + ty;
    
    int x_in = x_out - KERNEL_RADIUS;
    int y_in = y_out - KERNEL_RADIUS;
    float res = 0.0;
    float templ_res = 0.0;
    float img_res = 0.0;
    if((x_in>=0) && (x_in<x_size) && (y_in>=0) && (y_in<y_size))
    {
        Nds[ty][tx] = A[y_in*x_size + x_in];
    }
    else
    {
        Nds[ty][tx] = 0.0;
    }
    __syncthreads();

    
    if( (tx<TILE_WIDTH) && (ty<TILE_WIDTH) && (x_out<x_size) && (y_out<y_size) )
    {
        for( int idx_y=0; idx_y<KERNEL_WIDTH; idx_y++ )
        {
            for( int idx_x=0; idx_x<KERNEL_WIDTH; idx_x++ )
            {
                templ_res += pow(deviceKernel[idx_y*KERNEL_WIDTH+idx_x],2);
                img_res += pow(Nds[ty+idx_y][tx+idx_x],2);
                res += Nds[ty+idx_y][tx+idx_x] * deviceKernel[idx_y*KERNEL_WIDTH+idx_x];
                //printf("%f\n",res);
  
            }
        }
        
        __syncthreads();
        if((x_out<x_size) && (y_out<y_size))
        {
            //printf("here\n");
            B[y_out*x_size + x_out] = res/sqrt(templ_res*img_res);
        }
    }
   
}
///////////////////////////////////////////////////////////////////
/* cuda_tp_img
 *      Description: This program use for preparation step for the 
 *                   cuda kernel   
 *      Input: templates number    
 *      Output: 0 -- success, -1 -- failure
 *      
 * 
*/
///////////////////////////////////////////////////////////////////
int cuda_tp_img(int template_num)
{
    int x_size = gray_img.cols;
    int y_size = gray_img.rows;
    int tmp_x_size = KERNEL_WIDTH;//templs[0].cols;
    int tmp_y_size = KERNEL_WIDTH;//templs[0].rows;
    cout << x_size << " " << y_size << endl;
    cout << tmp_x_size << " " << tmp_y_size << endl;
    int img_size = x_size * y_size;
    int tmpl_size = tmp_x_size * tmp_y_size;
    
    //allocate a space to store the image intensity
    float* host_img = (float*) malloc(sizeof(float)*img_size);
    float* host_templ = (float*) malloc(sizeof(float)*tmpl_size);

    float* device_img_input;
    float* device_img_output;
    float gpu_out[307200];
    
    for(int y=0; y<y_size; y++)
    {
        for(int x=0; x<x_size; x++)
        {
            Scalar intensity = gray_img.at<uchar>(y,x);
            host_img[y*x_size+x] = intensity.val[0]/100;
            //cout << host_img[y*x_size+x]<<" ";
        }
        //cout << endl;
        
    }
    for(int y=0; y<tmp_y_size; y++)
    {
        for(int x=0; x<tmp_x_size; x++)
        {
            Scalar intensity = templs[0].at<uchar>(y,x);
            host_templ[y*tmp_x_size+x] = intensity.val[0]/100;
            //cout << host_templ[y*tmp_x_size+x]<<" ";
        }
        //cout << endl;        
    }
    //allocate memory in cuda global memory
    cudaMalloc( (void**)&device_img_input, img_size*sizeof(float) );
    cudaMalloc( (void**)&device_img_output, img_size*sizeof(float) );

    cout << cudaMemcpy( device_img_input, host_img, img_size*sizeof(float), cudaMemcpyHostToDevice) << endl;
    cout << cudaMemcpyToSymbol( deviceKernel, host_templ, tmpl_size*sizeof(float)) << endl;

    //assign blocks and threads
    dim3 Dimblock(BLK_SIZE, BLK_SIZE, 1);
    dim3 DimGrid(((TILE_WIDTH+x_size)-1/TILE_WIDTH), ((TILE_WIDTH+y_size)-1/TILE_WIDTH),1);
    //calling the convolution gpu function
    conv2d <<< DimGrid, Dimblock >>>( device_img_input, device_img_output, x_size, y_size);
    cudaDeviceSynchronize();
    
    cout << cudaMemcpy( gpu_out, device_img_output, img_size*sizeof(float), cudaMemcpyDeviceToHost) << endl;
    float res = 0;
    int x_pos, y_pos;
    for(int y=0; y<y_size; y++)
    {
        for(int x=0; x<x_size; x++)
        {
           cout << gpu_out[y*x_size+x] << " ";
           if(gpu_out[y*x_size+x]>res)
           {
                res = gpu_out[y*x_size+x];
                x_pos = x;
                y_pos = y;
           }
        }  
        cout << endl;      
    }
    //cout << x_pos << " " << y_pos << " " << res << endl;
    //cout << gpu_out[59*x_size+102] << endl;
    rectangle(gray_img, Point(x_pos-12,y_pos-12), Point(x_pos+12,y_pos+12), Scalar(0,255,0 ), 2, 4);
    cudaFree(device_img_input);
    cudaFree(device_img_output);
    free(host_img);
    free(host_templ);
    return 0;
}


int main(int argc, char*argv[])
{
    //first check how many argumnents are input
    int template_num;
    if(argc < 2)
    {
        cout << "wrong input" << endl;
        cout << "Usage: ./cuda_template_matching [image] 'tempalte1, template2, template3, ..." << endl;
        return -1;
    }
    if(argc == 2)
    {
        cout << "there is no templates need to match" << endl;
        return -1;
        
    }
    template_num = argc - 2;
    img = imread( argv[1], 1);
    //convert the image to gray scale in order to only have one pointer
    cvtColor(img, gray_img, CV_BGR2GRAY);
    //loading the template into the program
    for( unsigned int i = 0; i<template_num; i++)
    {
        Mat curr_tmpl = imread(argv[2+i],1);
        cvtColor(curr_tmpl, templs[i], CV_BGR2GRAY);
        //cout << templs[i].cols << " " << templs[i].rows << endl;
        //imshow("curr tmpl", templs[i]);
        //waitKey(0);

    }
    cuda_tp_img(template_num);
    imshow("img", gray_img);
    waitKey(0);
    return 0;
}
