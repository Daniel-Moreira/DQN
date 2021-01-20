FROM tensorflow/tensorflow:latest-gpu-jupyter

ENV APP_HOME="/src"

COPY requirements.txt ${APP_HOME}/

RUN pip3 install -r ${APP_HOME}/requirements.txt
# RUN pip3 install gym[atari]
RUN apt-get install python-opengl -y
RUN apt-get install xvfb mesa-utils -y
RUN apt-get install ffmpeg -y

COPY roms /roms

ADD startup_scrip.sh /usr/local/bin/startup_script.sh 
RUN chmod 777 /usr/local/bin/startup_script.sh

WORKDIR ${APP_HOME}
COPY space_invaders space_invaders/
COPY src .

RUN python3 -m retro.import /roms

ENTRYPOINT [ "/usr/local/bin/startup_script.sh" ]