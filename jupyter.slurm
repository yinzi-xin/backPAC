#!/bin/bash

#SBATCH --job-name=jupyter_notebook
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=interactive
#SBATCH --time=2:00:00
#SBATCH --gres=gpu
#SBATCH --output=logs/%x_%j.out

unset XDG_RUNTIME_DIR
module load uv
module load CUDA/12.3.2

echo "Running the notebook on $(hostname)"
IPADDR=$(hostname -i)

cd /home/xiny1/backPAC/
unset LD_LIBRARY_PATH # https://github.com/jax-ml/jax/issues/28929

source .venv/bin/activate
port=8988

SALT="$(head /dev/urandom | tr -dc 'A-Z-a-z-0-9' | head -c 24;echo;)"
password="13e17121t2"
PASSWORD_SHA="$(echo -n "${password}${SALT}" | openssl dgst -sha256 | awk '{print $NF}')"
echo "-------------------------"
echo "Log in using the password: $password"
echo "-------------------------"

# the jupyter server config file
export CONFIG_FILE="${PWD}/config.py"

(
umask 077
cat > "${CONFIG_FILE}" << EOL
c.ServerApp.ip = '${IPADDR}'
c.ServerApp.port = ${port}
c.ServerApp.port_retries = 1
c.ServerApp.password = u'sha256:${SALT}:${PASSWORD_SHA}'
c.ServerApp.base_url = '/node/$(hostname)/${port}/'
c.ServerApp.open_browser = False
c.ServerApp.allow_origin = '*'
c.ServerApp.root_dir = '${HOME}/mplc-inverse-design/'
c.ServerApp.disable_check_xsrf = True
EOL
)

echo "#### Starting the JupyterLab server"
set -x
jupyter lab --config="${CONFIG_FILE}"
echo "#### Terminated JupyterLab server. Done"
