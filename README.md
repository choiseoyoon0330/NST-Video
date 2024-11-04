# NST-Video

This project applies [AdaIN](https://arxiv.org/abs/1703.06868) (Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization) to video frames, allowing users to create artistic videos by transferring the style of a static image to a video.

## Project Structure
- **`/data`**: Contains the images and videos for NST application.
- **`/models`**: Stores pre-trained deep learning models.
- **`/results`**: Contains the generated styled videos.
- **`nst_video.py`**: Main pipeline code for applying NST.
- **`utils.py`**: Utility functions for data handling and model loading.

## Usage

1. Prepare a style image and content video. Place the style image and content video in the `/data` folder.

2. Apply NST model to the video:
   ```bash
   python nst_video.py --content_video path/to/content_video.mp4 --style_image path/to/style_image.jpg --output path/to/output_video.mp4
   ```
3. Options
  - **`--content_video`**:  Path to the content video
  - **`--style_image`**:  Path to the style image
  - **`--output`**:  Path to save the output video

## Example
- Style Image: **`data/style_image.jpg`**
- Content Video: **`data/content_video.mp4`**

Command:  
   ```bash
   python nst_video.py --content_video data/content_video.mp4 --style_image data/style_image.jpg --output results/stylized_video.mp4
   ```

## Results

| Content Video              | Style Image               | Styled Video                   |
|----------------------------|---------------------------|--------------------------------|
| ![Content Video](https://github.com/user-attachments/assets/c2185317-a53c-4575-97f8-163cfe9a39de) | <img src="https://github.com/user-attachments/assets/9a9e8259-cc2b-4cf2-bda5-0724a34f72f3" width="650" > | ![Styled Video](https://github.com/user-attachments/assets/ac808c0c-8fa6-4242-98d2-34f432285e7a) |
| ![Content Video](https://github.com/user-attachments/assets/02dcd40a-b44f-461f-a176-d8b37b51d442) | <img src="https://github.com/user-attachments/assets/293d7432-23de-4498-bdbe-556190100a4e" width="650" > | ![Styled Video](https://github.com/user-attachments/assets/796e3e29-a558-4542-8f3c-e75bf783e8c3) |

## Model Training

To train your own model, use the **`train.py`** script.

```bash
python train.py --style_image path/to/style_image.jpg
```

## References

- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)


<!-- # NST_Video
Neural Style Transfer for Video

Applied  to the video
-->

