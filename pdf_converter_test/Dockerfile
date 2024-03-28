ARG POPPLER_VERSION="24.03.0"
ARG POPPLER_DATA_VERSION="0.4.12"
ARG OPENJPEG_VERSION="2.4.0"

FROM public.ecr.aws/lambda/python:3.9-x86_64
RUN /var/lang/bin/python -m pip install --upgrade pip
RUN python -m pip install boto3>1.28.57 pdf2image amazon-textract-idp-cdk-manifest --target "${LAMBDA_TASK_ROOT}"

ARG POPPLER_VERSION
ARG POPPLER_DATA_VERSION
ARG OPENJPEG_VERSION

WORKDIR /root

RUN yum update -y
RUN yum install -y \
   cmake \
   cmake3 \
   fontconfig-devel \
   gcc \
   gcc-c++ \
   gzip \
   libjpeg-devel \
   libpng-devel \
   libtiff-devel \
   make \
   tar \
   xz \
   zip

RUN curl -o poppler.tar.xz https://poppler.freedesktop.org/poppler-${POPPLER_VERSION}.tar.xz
RUN tar xf poppler.tar.xz
RUN curl -o poppler-data.tar.gz https://poppler.freedesktop.org/poppler-data-${POPPLER_DATA_VERSION}.tar.gz
RUN tar xf poppler-data.tar.gz
RUN curl -o openjpeg.tar.gz https://codeload.github.com/uclouvain/openjpeg/tar.gz/refs/tags/v${OPENJPEG_VERSION}
RUN tar xf openjpeg.tar.gz

WORKDIR poppler-data-${POPPLER_DATA_VERSION}
RUN make install

WORKDIR /root
RUN mkdir openjpeg-${OPENJPEG_VERSION}/build
WORKDIR openjpeg-${OPENJPEG_VERSION}/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make
RUN make install

WORKDIR /root
RUN mkdir poppler-${POPPLER_VERSION}/build
WORKDIR poppler-${POPPLER_VERSION}/build
RUN cmake3 .. -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_STATIC_LIBS=OFF
RUN make
RUN make install


WORKDIR /root
RUN mkdir -p package/{lib,bin,share}
RUN cp -d /usr/lib64/libexpat* package/lib
RUN cp -d /usr/lib64/libfontconfig* package/lib
RUN cp -d /usr/lib64/libfreetype* package/lib
RUN cp -d /usr/lib64/libjbig* package/lib
RUN cp -d /usr/lib64/libjpeg* package/lib
RUN cp -d /usr/lib64/libpng* package/lib
RUN cp -d /usr/lib64/libtiff* package/lib
RUN cp -d /usr/lib64/libuuid* package/lib
RUN cp -d /usr/lib64/libz* package/lib
RUN cp -rd /usr/local/lib/* package/lib
RUN cp -rd /usr/local/lib64/* package/lib
RUN cp -d /usr/local/bin/* package/bin
RUN cp -rd /usr/local/share/poppler package/share
COPY package/* ${LAMBDA_TASK_ROOT}/

# Copy function code
COPY app/* ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.lambda_handler" ]
