FROM ground-texture-slam:1.0.4
USER root
# Install dependencies, including the ability to give sudo rights.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    less \
    sudo \
    && rm -rf /var/lib/apt/lists/*
# Give the "user" user sudo rights.
RUN echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user
USER user
