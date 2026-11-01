ARG PYTHON_IMAGE=python:3.14.7-slim-trixie
FROM ${PYTHON_IMAGE} AS ompl
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG OMPL_VERSION=2.0.2
ARG OMPL_SHA256=b9155e243e555b04b01f8e4ff7e21a48582edbab9c300a611289655778bcd29d
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build pkg-config curl ca-certificates \
    libboost-dev libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev \
    libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/ompl.tar.gz https://github.com/ompl/ompl/releases/download/${OMPL_VERSION}/ompl-${OMPL_VERSION}.tar.gz \
 && echo "${OMPL_SHA256}  /tmp/ompl.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/ompl.tar.gz -C /tmp
ARG BUILD_JOBS=3
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
RUN pip wheel --no-cache-dir --no-deps -w /wheels /tmp/ompl-${OMPL_VERSION}/py-bindings
RUN mkdir -p /wheels/lib \
 && pip install --no-cache-dir --no-deps --target /tmp/ompl-probe /wheels/ompl-*.whl \
 && find /tmp/ompl-probe -name '*.so' -exec ldd {} + \
    | awk '/libboost/ {print $3}' | sort -u | xargs -r -I{} cp -L {} /wheels/lib/

FROM ${PYTHON_IMAGE} AS dependencies
COPY --from=ompl /wheels /tmp/wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 libgomp1 libgl1 libglib2.0-0 procps openssh-client \
 && rm -rf /var/lib/apt/lists/* \
 && cp /tmp/wheels/lib/* /usr/local/lib/ \
 && ldconfig \
 && pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir /tmp/wheels/ompl-*.whl \
    drake==1.57.0 numpy==2.5.3 matplotlib==3.11.2 pyyaml==6.0.3 \
    stable-baselines3==2.9.0 gymnasium==1.3.0 \
 && rm -rf /tmp/wheels \
 && python -m pip check \
 && python -c "import ompl.base, pydrake, torch, stable_baselines3; from pydrake.solvers import SnoptSolver; assert SnoptSolver().available()"
ENV PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM dependencies AS application
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin planar \
 && mkdir -p /app/runs /app/policy /app/banks /app/data/pose-maps \
 && chown -R planar:planar /app
WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md pyproject.toml ./
RUN chmod -R a+rX src scripts README.md pyproject.toml
USER planar
EXPOSE 4880 4881 7800-7820 7840-7860
VOLUME ["/app/runs"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["python", "-c", "from urllib.request import urlopen; assert all(b'Physics workspace' in urlopen(f'http://127.0.0.1:{port}/', timeout=2).read() for port in (4880, 4881))"]
CMD ["python", "-m", "dashboard", "--host", "0.0.0.0"]
