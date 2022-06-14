###############################################################################
# Info:                                                                       #
#    This Dockerfile is used to build a Docker image for your submission.     #
#    Participants will not have access to the base submission image as it     #
#    contains the evaluation track.                                           #
#                                                                             #
# Build:                                                                      #
#    Your submission will be built as follows:                                #
#    $ docker build -t ${NAME}:${TAG} -f ${DOCKERFILE} .                      #     
#                                                                             #
#    where {NAME} will be associated with the participant's name and {TAG}    #
#    will be a unique identifier for the submission                           #
#                                                                             #
# Base image file structure:                                                  #
#    The base submission image contains the entire l2r directory:             #
#    TODO (add link) <github link>                                            #
#                                                                             #
# Submission files:                                                           #
#    Your submission should contain all files in a single directory and       #
#    should contain, at minimum, eval.py and conf.yml. You may also include   #
#    additional files such as:                                                #
#      - requirements.txt (for pip install)                                   #
#      - environment.yml (for conda virtual environments)                     #
#      - venv directory (with venv/bin/activate, for a complete venv)         #
#      - model weights                                                        #
#      - encoders & weights                                                   #
#      - etc.                                                                 #
#                                                                             #
###############################################################################

FROM ubuntu:18.04

# Add submission to the image
COPY submission.zip /home

WORKDIR /home/l2r
