////////////////////////////////////////////////////////////////////////////////
///                             File Name: a.cpp                             ///
///                          Author: Huaxiao Liang                           ///
///                         Mail: 1184903633@qq.com                          ///
///                         02/15/2025-Sat-13:41:42                          ///
////////////////////////////////////////////////////////////////////////////////

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>

int main() {
    int width, height, channels;

    // Load a JPEG image file into memory
    unsigned char *image_data = stbi_load("/home/banana/cpp_workspace/OpenCL/image_filtering/input_img.jpg", &width, &height, &channels, 0);

    if (image_data == nullptr) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    std::cout << "Loaded image with width: " << width << ", height: " << height << ", channels: " << channels << "\n";

    // Free the image data when done
    stbi_image_free(image_data);

    return 0;
}

