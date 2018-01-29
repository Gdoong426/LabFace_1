# LabFace_1
Use Opencv and dlib to create a face detection and recognizer application to find if a detected face is unknown face or one of our members.

## Workflow
1. Use dlib-face_landmark_detector to find faces in the current frames, and crop and align those faces for latter training process.
2. When the amount of faces are enough for training, train an opencv_eigenface/fishface descriptor and save it into file.
3. When actual running, first use dlib face_landmark to crop and align detected faces images, and then use the stored face descriptor to recognize those faces.

## Todo
1. Tune the parameter in opencv_eigenface params
2. Save detected faces after face recognition and seperate them into different folder.
