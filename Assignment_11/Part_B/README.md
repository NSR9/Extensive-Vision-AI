## Problem Statement:

Training Custom Dataset on Colab for YoloV3
1. Refer to this Colab File:  [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
2. Refer to this [GitHub  Repo](https://github.com/theschoolofai/YoloV3)
3. Download this [dataset](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing) (Links to an external site.). This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
    - class names are in custom.names file. 
    - you must follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
    - Once you add your additional 100 images, train the model
4. Once done:
    - [Download](https://www.y2mate.com/en19) a very small (~10-30sec) video from youtube which shows your classes. 
    - Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to extract frames from the video. 
    - Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
    - Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 

            python detect.py --conf-three 0.3 --output output_folder_name

    - Use  [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to convert the files in your output folder to video
    - Upload the video to YouTube. 
    - Also run the model on 16 images that you have collected (4 for each class)

## Annotating Custom Dataset
* For Custom Dataset
    1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
    2. Follow the steps in the ReadME of the cloned repo. 

## Parameters changed:

1. Use custom dataset of 4000 imaages downloaded from the above link. Added 100 more images by annotaing using the Annotaion tool 
2. Custom Classes:
    - hardhat
    - vest
    - mask
    - boots
2. filters=255
3. classes=80
4. burn_in to 100
5. max_batches to 5000
6. steps to 4000,4500

## Results: 

### Youtube Video Link:
[Object Detection Video](https://youtu.be/DwS8OtbGy9I)

### Social Media Link:
[LinkedIn Link]

### YoloV3 on Custom data:

Here are a few of the images annotated using the YoloV3 trained on Custom Dataset Provided:
| Hardhat | Vest | Mask | Boots |
|--|--|--|--|
| <img src="https://user-images.githubusercontent.com/51078583/126872938-656e8b99-b40a-458b-95cc-94483598c962.png" alt="Girl in a jacket" width="180" height="200">| ![image](https://user-images.githubusercontent.com/51078583/126873027-4b229e5d-9105-46b9-aeea-ad635bcd8ae4.png) | ![image](https://user-images.githubusercontent.com/51078583/126872995-69e9f5c6-27d9-4b5f-86e8-169018d7791b.png) | <img src="https://user-images.githubusercontent.com/51078583/126872954-d2643fd8-6084-42ff-b4da-c818201f3a81.png" alt="Girl in a jacket" width="200" height="200"> |
| <img src="https://user-images.githubusercontent.com/51078583/126873041-4b9c5ab7-451e-4f86-a673-d55ab7fe0bcc.png" alt="Girl in a jacket" width="200" height="200"> | <img src="https://user-images.githubusercontent.com/51078583/126873081-39fe8a9c-2d5d-46e6-8d18-5a156344b9af.png" alt="Girl in a jacket" width="200" height="200"> | <img src="https://user-images.githubusercontent.com/51078583/126873011-9848c060-9c56-4c30-ad34-f1bdb838dd29.png" alt="Girl in a jacket" width="180" height="200"> | <img src="https://user-images.githubusercontent.com/51078583/126872974-4baf6ad9-ae93-4810-82b5-4eb80ffffe96.png" alt="Girl in a jacket" width="140" height="200"> |
| ![image](https://user-images.githubusercontent.com/51078583/126873048-ce297476-d9ba-4eac-858b-cec49c3b0cc7.png) |  | ![image](https://user-images.githubusercontent.com/51078583/126873020-e8e86242-1ac7-40d1-8c54-f2d535f2daab.png) |  |
|  |  | ![image](https://user-images.githubusercontent.com/51078583/126873074-75e4a4eb-8b47-4ca1-9267-f3bf22397f98.png) |  |

## References
[Yolo V3 Sample](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
[SchoolOfAI YoloV3 Repo](https://github.com/theschoolofai/YoloV3)
[FFMPEG](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)

## Contirbutors:

1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
