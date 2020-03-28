FROM vinay0410/tectonic-image:latest

RUN apt-get install fonts-font-awesome

COPY build.sh /build.sh

ENTRYPOINT ["/build.sh"]
