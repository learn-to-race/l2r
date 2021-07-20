
Default Camera Configuration
============================

Overview
--------

The agent has access to camera sensor data which are raw pixel values (with bounds of 0 and 255). The default vehicle configuration includes 8 cameras with names:

* CameraFrontRGB (CameraFrontSegm)
* CameraLeftRGB (CameraLeftSegm)
* CameraRightRGB (CameraRightSegm)
* CameraBirdsEyeRGB (CameraBirdsEyeSegm)

To include any subset of these cameras, simply include the camera's name and parameters under ``env_kwargs.cameras``. For example:

.. code-block:: yaml

   env_kwargs:
      cameras:
         CameraFrontSegm:
            Addr: 'tcp://0.0.0.0:9008'
            Format: SegmBGR8
            Width: 512
            Height: 384
            bAutoAdvertise: True

Modifying Cameras
-----------------

The camera is flexible in terms of both the field-of-view angle, dimensions of the images, type of camera, and position. To modify the camera, simply change the parameters:

.. code-block:: yaml

   GenericCamera:                 # Camera name which will be a key of observation dictionary
      Addr: 'tcp://0.0.0.0:9008'  # Address camera will publish to
      Format: ColorBGR8           # ColorBGR8, SegmBGR8, HdrBGR8
      FOVAngle: 120               # modifiable field of view parameter, in degrees
      Width: 256                  # modifiable image width parameter, in number of pixels
      Height: 256                 # modifiable image height parameter, in number of pixels
      bAutoAdvertise: True

To set the environment to vision-only, set the ``multimodal`` parameter to ``False`` in the ``configs/params.yaml`` file.

.. code-block:: yaml

   env_kwargs:
      multimodal: False
      ...

Creating Cameras
----------------

See `Creating Custom Sensor Configurations <sensors.html>`_
