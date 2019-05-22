#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <random>
#include <string>
#include <ctime>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

//constants
#define TRAINING_SET_SIZE 60000
#define TEST_SET_SIZE 10000
#define COLS 28
#define ROWS 28
#define IMG_MAGIC_NUM 0x00000803
#define LABEL_MAGIC_NUM 0x00000801
#define NUM_LABELS 60000
#define NUM_NEURONS 1024
#define EPOCH_SIZE 100
#define BATCH_SIZE 100

//All the integers in the files are stored in the MSB first (high endian) format
void toLittleEndian(int &num){
    num = (0xFF&(num >> 24))      |
          (0xFF00&(num >> 8))     |
          (0xFF0000&(num << 8))   |
          (0xFF000000&(num << 24));
}

void read_images(const std::string &file_name, float*** (&imgs)){
    int fd;
    fd = open(file_name.c_str(), O_RDONLY);
    assert(fd >= 0);

    int rv, magic_num, num_imgs, num_cols, num_rows;

    rv = read(fd, &magic_num, 4);
    assert(rv == 4);
    //change endianess
    toLittleEndian(magic_num);
    assert(magic_num == 0x803);

    rv = read(fd, &num_imgs, 4);
    assert(rv == 4);
    //change endianess
    toLittleEndian(num_imgs);

    rv = read(fd, &num_rows, 4);
    assert(rv == 4);
    //change endianness
    toLittleEndian(num_rows);

    rv = read(fd, &num_cols, 4);
    assert(rv == 4);
    //change endianness
    toLittleEndian(num_cols);

    imgs = new float**[num_imgs]();
    for(int i = 0; i < num_imgs; i++){
        imgs[i] = new float*[num_rows]();

        //read whole image at once to minimize IO since that takes time
        unsigned char tmp_img[num_rows][num_cols];
        rv = read(fd, tmp_img, num_rows*num_cols);
        assert(rv == num_rows*num_cols);

        for(int r = 0; r < num_rows; r++){
            imgs[i][r] = new float[num_cols]();
            for(int c = 0; c < num_cols; c++){
                imgs[i][r][c] = double(tmp_img[r][c])/127.5 - 1;
            }
        }

    }
    rv = close(fd);
    assert(rv == 0);
}

void read_labels(const std::string &file_name, unsigned char* (&labels)){
    int fd;
    fd = open(file_name.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic_num, num_labels;

    int rv = read(fd, &magic_num, 4);
    assert(rv == 4);
    //change endianess
    toLittleEndian(magic_num);

    rv = read(fd, &num_labels, 4);
    assert(rv == 4);
    //change endianess
    toLittleEndian(num_labels);

    labels = new unsigned char[num_labels]();
    rv = read(fd, labels, num_labels);
    for(int i = 0; i < num_labels; i++){
        //all labels are 0-9
        //assert(labels[i] >= 0 && labels[i] <= 9);
    }
    rv = close(fd);
    assert(rv == 0);

}

void generateWeights(float*** (&ilw), float*** (&ild), float** (&fclw), float** (&fcld)){

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 8493;
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0, 1.0);

    ilw = new float**[NUM_NEURONS]();
    ild = new float**[NUM_NEURONS]();
    fclw = new float*[(int) NUM_NEURONS/EPOCH_SIZE]();
    fcld = new float*[(int) NUM_NEURONS/EPOCH_SIZE]();

    for(int n = 0; n < NUM_NEURONS; n++){
        ilw[n] = new float*[ROWS]();
        ild[n] = new float*[ROWS]();
        for(int r = 0; r < ROWS; r++){
            ilw[n][r] = new float[COLS]();
            ild[n][r] = new float[COLS]();
            for(int c = 0; c < COLS; c++){
                //normal_distribution represents unbownded distribution, divide by sqrt(N)
                ilw[n][r][c] = distribution(generator) / sqrt(n);
                //initially weights are 0
                ild[n][r][c] = 0;
            }
        }
    }

    for(int i = 0; i < (int) NUM_NEURONS/EPOCH_SIZE; i++){
        fclw[i] = new float[NUM_NEURONS]();
        fcld[i] = new float[NUM_NEURONS]();
        for (int n = 0; n < NUM_NEURONS; n++){
            fclw[i][n] = distribution(generator) / sqrt((int) NUM_NEURONS/EPOCH_SIZE);
            fcld[i][n] = 0;
        }
    }
}

//based on softmax from prof. Chiu's examples
float* softmax(float *in){
    // Use identity softmax(x) == softmax(x - C)
    const auto C = *std::max_element(in, in+((int) NUM_NEURONS/EPOCH_SIZE));
    //same length as in
    float* out = new float[(int) NUM_NEURONS/EPOCH_SIZE];
    float sum;
    for(size_t i = 0; i < (int) NUM_NEURONS/EPOCH_SIZE; i++){
        out[i] = std::exp(in[i] - C);
        sum += out[i];
    }
    /*
    for(size_t i = 0; i < (int) NUM_NEURONS/EPOCH_SIZE; i++){
        out[i] = out[i]/sum;
    }
    */
    std::transform(out, out + ((int) NUM_NEURONS/EPOCH_SIZE), out, [sum](float e) {return e/sum;});

    return out;

}

float* softmax_ds(float* out, float* us){
    float* sm_ds = new float[(int) NUM_NEURONS/EPOCH_SIZE]();
    for(size_t i = 0; i < (int) NUM_NEURONS/EPOCH_SIZE; i++){
        for(size_t j = 0; j < (int) NUM_NEURONS/EPOCH_SIZE; j++){
            if( i == j) {
                sm_ds[j] += (out[i]*(1 - out[j])) * us[i];
            } else {
                sm_ds[j] += (-out[j]*out[i])*us[i];
            }
        }
    }
    return sm_ds;
}

int main(int argc, char** argv){
    if(argc != 3){
        std::cerr << "Wrong number of inputs. Usage: ./parallelNN <images> <labels>" << std::endl;
        exit(1);
    }

    //read training
    static float ***training_images;
    static unsigned char *training_labels;
    read_images(std::string(argv[1]), training_images);
    read_labels(std::string(argv[2]), training_labels);

    float ***input_layer_w, ***input_layer_deriv;
    float **fully_connected_layer_w, **fully_connected_layer_deriv;

    generateWeights(input_layer_w, input_layer_deriv, fully_connected_layer_w, fully_connected_layer_deriv);

    //First fully connected layer
    float *first_layer = new float[NUM_NEURONS]();
    float *first_layer_deriv = new float[NUM_NEURONS]();

    //Second fully connected layer
    float *second_layer = new float[(int)NUM_NEURONS/EPOCH_SIZE]();
    float *second_layer_deriv = new float[NUM_NEURONS]();

    //Softmax layer
    float *soft_max_layer = new float[(int)NUM_NEURONS/EPOCH_SIZE]();
    float *soft_max_layer_deriv = new float[(int)NUM_NEURONS/EPOCH_SIZE]();

    //Cross-entropy layer
    float* cross_ent_layer = new float[(int)NUM_NEURONS/EPOCH_SIZE]();

    //bias nodes
    float *first_layer_bias = new float[NUM_NEURONS]();
    float *first_layer_bias_deriv = new float[NUM_NEURONS]();
    float *second_layer_bias =  new float[(int)NUM_NEURONS/EPOCH_SIZE]();
    float *second_layer_bias_deriv = new float[(int)NUM_NEURONS/EPOCH_SIZE]();

    std::srand(std::time(0));

    for(int e = 0; e < EPOCH_SIZE; e++){
        for(int j = 0; j < EPOCH_SIZE; j++){

            //FORWARD

            //initialize values
            int correct = 0, total = 0;

            //loop through images in batch
            for(int i = 0; i < BATCH_SIZE; i++){
                for(int k = 0; k < (int) NUM_NEURONS/EPOCH_SIZE; k++){
                    cross_ent_layer[k] = 0;
                }
                //current label and img displaced by i (the images already processed)
                int current_label = (int) training_labels[EPOCH_SIZE*j + i];
                float** current_image = training_images[EPOCH_SIZE*j + i];

                float learning_rate = BATCH_SIZE / 1000;
                for(int n = 0; n < NUM_NEURONS; n++){
                    float temp_result = 0;
                    //dropout rate of 0.4%
                    if(std::rand() % 1000 < 4){
                        first_layer[n] = 0;
                    } else{
                        for(int r = 0; r < ROWS; r++){
                            for(int c = 0; c < COLS; c++){
                                //calculate results of the first layer
                                temp_result += input_layer_w[n][r][c] * current_image[r][c];
                            }
                        }
                        //ReLU
                        temp_result += first_layer_bias[n];
                        if(temp_result < 0){
                            first_layer[n] = 0;
                        } else{
                            first_layer[n] = temp_result;
                        }
                    }
                }

                for(int k = 0; k < (int) NUM_NEURONS/EPOCH_SIZE; k++){
                    for(int n = 0; n < NUM_NEURONS; n++){
                        second_layer[k] += fully_connected_layer_w[k][n] * first_layer[n];
                    }
                    second_layer[k] += second_layer_bias[k];
                }

                soft_max_layer = softmax(second_layer);
                if(std::distance(soft_max_layer, std::max_element(soft_max_layer, soft_max_layer+(int) NUM_NEURONS/EPOCH_SIZE)) == current_label){
                    correct++;
                }
                total++;

                cross_ent_layer[current_label] = -1 / soft_max_layer[current_label];

                //BACK-PROPAGATION

                soft_max_layer_deriv = softmax_ds(soft_max_layer, cross_ent_layer);

                for(int k = 0; k < (int) NUM_NEURONS/EPOCH_SIZE; k++){
                    for(int n = 0; n < NUM_NEURONS; n++){
                        second_layer_deriv[n] = 0;
                    }
                    for(int n = 0; n < NUM_NEURONS; n++){
                        fully_connected_layer_deriv[k][n] += ((first_layer[n] * soft_max_layer_deriv[k]) / BATCH_SIZE);
                        second_layer_deriv[n] += fully_connected_layer_w[k][n] * soft_max_layer_deriv[k];
                    }
                    second_layer_bias_deriv[k] = soft_max_layer_deriv[k] / BATCH_SIZE;
                }

                for(int n = 0; n < NUM_NEURONS; n++){
                    for(int r = 0; r < ROWS; r++){
                        for(int c = 0; c < COLS; c++){
                            input_layer_deriv[n][r][c] +=  (current_image[r][c] * second_layer_deriv[n])/BATCH_SIZE;
                        }
                    }
                    first_layer_bias_deriv[n] = second_layer_deriv[n] / BATCH_SIZE;
                }

                //UPDATE WEIGHTS

                for(int n = 0; n < NUM_NEURONS; n++){
                    for(int r = 0; r < ROWS; r++){
                        for(int c = 0; c < COLS; c++){
                            input_layer_w[n][r][c] -=  input_layer_deriv[n][r][c] * learning_rate;
                            input_layer_deriv[n][r][c] = 0;
                        }
                    }
                    first_layer_bias[n] -= first_layer_bias_deriv[n] * learning_rate;
                    first_layer_bias_deriv[n] = 0;
                }

                for(int k = 0; k < (int) NUM_NEURONS/EPOCH_SIZE; k++){
                    for(int n = 0; n < NUM_NEURONS; n++){
                        fully_connected_layer_w[k][n] -= fully_connected_layer_deriv[k][n] * learning_rate;
                        fully_connected_layer_deriv[k][n] = 0;
                    }
                    second_layer_bias[k] -= second_layer_bias_deriv[n] * learning_rate;
                    second_layer_bias_deriv[k] = 0;
                }
            }
            if(j % 5 == 0){
                printf("ACCURACY = %f\n", correct / total);
            }

        }
    }

    return 0;

}























//
