# TripleD
Real Time - Driver Drowsiness Detection using CNN

Operating a motor vehicle when sleepy is known as Drowsy driving. It increases risk of accidents - leading cause of death toll in roads. Each year, drowsy driving accounts for about 100,000 crashes, 71,000 injuries, and 1,550 fatalities, according to the National Safety Council (NSC). Reduced sleep length among adults is a factor for drowsiness. Awareness on causes, consequences, and prevention of drowsy driving is one of the building blocks to avoid similar situations.\
Tips to avoid drowsy driving:\
&nbsp; Stop and let your body rest\
&nbsp; Avoid fiddling with AC & radio to stay awake\
&nbsp; Utilize caffeine to stay alert (keep it moderate)

Steps involved in the project:
1. Face and Eye Detection :
Detects the face and then eyes within the face using Haar cascades.
2. Eye Aspect Ratio (EAR) :
Computes the Eye Aspect Ratio to determine if the eyes are closed.
3. Drowsiness Alert :
If the EAR is below a threshold for a continuous period, the system triggers an alert indicating potential drowsiness.
4. Beverage Recommendation :
Recommends beverages based on the drowsiness level detected.

What is EAR (Eye Aspect Ratio)?\
Eye closure is detected by estimating EAR from the ratio of distances between facial landmarks on eyes detected. The Eye Aspect Ratio, or EAR, is a scalar value that responds, for  opening and closing the eyes. The more the EAR, the more widely eye is open. Decide a minimum EAR value and set this threshold to decide if the eye is closed or not. When EAR goes below threshold, drowsiness is confirmed and driver is alerted. It is an inexpensive and cost effective image processing technique.

## Datasets used:
[Shape predictor 68 face landmarks](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat):
* DAT file, initialize dlib's face detector
* create the facial landmark predictor
* loop over the face detections
* determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
* loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
* show the output image with the face detections + facial landmarks

[MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset): 
* A large-scale dataset of human eye images designed for computer vision tasks such as eye detection and blink detection. 
* Includes 84,898 images captured under various conditions.
* commonly used in computer vision and deep learning projects that focus on gaze detection, blink detection, and eye state classification
* varying resolutions of eye images cropped
* labeled with closed and open

[Haar Cascade files](https://github.com/anaustinbeing/haar-cascade-files):
* pre-trained classifiers used for object detection from images or video streams.
- Steps:
1) Haar-like feature identification
2) Feature extraction
3) Integral image
4) Sliding window
5) Cascade of classifiers
6) Output

## Drowsiness detection workflow:
- Webcam Feed (Live video captured frame by frame) 

- Face Detection (Finds the face and highlights it with a rectangle) “Haar cascade”

- Landmark Detection (Identifies eyes using key points) “Dlib Predictor”

- Eye Aspect Ratio (Analyzes eye openness to detect drowsiness)

- State Classification (Active, Drowsy, Sleeping)

- Alerts and Recommendations depending on the time of the day (Sound alarm, suggest coffee, etc.)

- Update GUI (Real-time status display and video feed)

## Utilities:
[<img src="main/utilities.png">](https://github.com/raisha-rfq/TripleD/edit/main/utilities.png)

## Drawbacks:
- Wearing glasses can lead to wrongful detection or failure to  recognise eye closure.
- Properly illuminated face is a necessity. This can be resolved by the use of infrared and LED lights especially during night.

## Overview of Libraries Used:
1. OpenCV: Captures live video → Processes video into frames → Video frame to grayscale
2. dlib: Detects the face in each video frame → Identifies 68 facial landmarks(Points 36–41: Left eye // Points 42–47: Right eye)
3. Tkinter: Creates the graphical interface (GUI) to display → (The live video feed,the driver’s current status (Active, Drowsy, or Sleeping),Recommendations like "Get some coffee!")
4. Pygame: Plays alarm sounds to alert the driver when they are drowsy or sleeping.
5. plyer: Sends notifications to warn the driver about their drowsy state (if notifications are enabled on the device).
6. PIL: Converts frames from the video capture into an image format that can be displayed in the tkinter GUI.
7. torchvision: Preprocess the eye images before passing them to the model for prediction using transforms function.
8. PyTorch: Defines model's neural network layers and loads it from a saved file.

## Model Architecture:
Input Layer: The CNN inputs eye images in grayscale resolution that are resized to 24x24 pixels.\
First Convolutional Layer (conv1): Input with grayscale image with one channel then applies 16 filters/kernels of size 3 x 3, using ReLU activation to introduce nonlinearity. After the convolution, applying a max-pooling using a 2 x 2 window to reduce dimensions to spatial (downsample half by feature map).\
Second Convolutional Layer (conv2): This layer takes up what the first layer produced and puts through 32 filters on its size 3x3, applies ReLU activation, and second max-pooling reduces the size more.\
Flatten Layer: The output of the second convolutional layer is flattened into a 1D vector to feed into the fully connected layers. The size of the flattened vector is 32 * 6 * 6 = 1152.\
Fully Connected Layer 1 (fc1): A fully connected layer with 128 neurons, which takes the flattened vector and applied ReLU.\
Output Layer (fc2): This layer produces a 2-dimensional vector of two possible eye states: Open (class 0) or Closed (class 1). This layer gives the raw scores for every class that will later be used in classification.
### Training Procedure:
Loss Function: This model uses CrossEntropyLoss to calculate the difference between the predicted class and the true class. Such loss function is suitable for multi-class classification problems such as ours.\
Optimizer: Adam optimizer is used for training, with a learning rate of 0.001. The optimizer updates the weights of the CNN according to the computed gradients in backpropagation.\
Training: The model is trained on multiple epochs (in the given code, 5 epochs were used) over a dataset of eye images. During each epoch, the model processes batches of images, computes the loss, and adjusts its weights to improve classification accuracy. After each epoch, the loss and accuracy are printed to monitor performance.
### Model Evaluation:
Accuracy: After training, the model's accuracy is tested on a test set. The given accuracy measures the performance of the model in terms of its ability to classify eye states - open or closed. Accurately, it demonstrates the model's performance over drowsiness detection.\
Test Accuracy: The code prints the test accuracy of the model, which indicates its ability to generalize to previously unseen data.\
Model Saving: After training, the trained model weights are saved as a.pth file, test_train_detect.pth, to allow reloading the model for future predictions without retraining.
