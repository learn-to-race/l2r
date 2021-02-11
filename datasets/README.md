# Observation Datasets

## About

Because of the importance of computer vision in the tasks we present, we have also included datasets, one for each racetrack, of 5000 observations. We generated these using the ``record_manually()`` method of ``envs/env.py`` (you are welcome to generate your own) and they consist of a handful of laps around the track driving in a zig-zag fashion to improve the diversity of the images from the camera.

These datasets may be useful for a variety of purposes such as:

- Pre-training an image encoder
- Training an image segmentation model
- Projecting the track boundaries onto the raw images
- etc.

## Format

Images were saved using ```numpy.savez_compressed()``` with arrays ``pose_data`` and ``image``. The images are in RGB format with a width of 448 pixels, height of 256 pixels, and a field of view of 90 degrees.

## Download

Please request access to the image datasets when you request access to the racing simulator.
