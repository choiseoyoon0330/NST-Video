# NST-Video

This project applies [AdaIN](https://arxiv.org/abs/1703.06868) (Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization) to video frames, allowing users to create artistic videos by transferring the style of a static image to a video. A simple approach was used by dividing the video into individual frames using OpenCV, then applying the style transfer model to each frame.

## Results

| Content Video              | Style Image               | Styled Video                   |
|----------------------------|---------------------------|--------------------------------|
| ![Content Video](https://github.com/user-attachments/assets/c2185317-a53c-4575-97f8-163cfe9a39de) | <img src="https://github.com/user-attachments/assets/9a9e8259-cc2b-4cf2-bda5-0724a34f72f3" width="600" > | ![Styled Video](https://github.com/user-attachments/assets/ac808c0c-8fa6-4242-98d2-34f432285e7a) |
| ![Content Video](https://github.com/user-attachments/assets/02dcd40a-b44f-461f-a176-d8b37b51d442) | <img src="https://github.com/user-attachments/assets/293d7432-23de-4498-bdbe-556190100a4e" width="600" > | ![Styled Video](https://github.com/user-attachments/assets/796e3e29-a558-4542-8f3c-e75bf783e8c3) |

## References

- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
