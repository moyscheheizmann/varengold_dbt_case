#!/usr/bin/env bash

# set color variables
GREEN='\e[32m'
GREEN_REVERSE='\e[32;7m'
RED='\e[31m'
YELLOW='\e[33m'
ELASTIC='\e[36;1;45m'
ENDC='\e[0m'

# initial print to terminal
echo ""
echo -e "${GREEN}Setting Things Up...${ENDC} ☕☕☕"
echo ""
sleep .5

# check for dbt packages
if [ ! -d "./transformation/dbt_packages" ]; then
    echo -e "${YELLOW}Installing dbt Packages...${ENDC} ☕☕☕"
    echo ""
    sleep .5
    ( cd transformation; dbt deps )
else
    echo -e "${GREEN}dbt packages are installed... ✔${ENDC}"
    echo ""
    sleep .5
fi

# setup terminal
apt update -y
apt install dos2unix wget unzip -y --no-install-recommends
wget https://github.com/duckdb/duckdb/releases/download/v0.9.1/duckdb_cli-linux-amd64.zip --directory-prefix="/root/"
unzip -o /root/duckdb_cli-linux-amd64.zip -d /usr/local/bin
rm /root/duckdb_cli-linux-amd64.zip
pip install -r ./transformation/requirements.txt
find .devcontainer/ -type f -print0 | xargs -0 dos2unix
cp .devcontainer/bash/.bashrc ~/.bashrc
cp .devcontainer/bash/.git-completion.bash ~/.git-completion.bash
cp .devcontainer/bash/.git-prompt.sh ~/.git-prompt.sh
source ~/.bashrc

# setup git
git config pull.rebase false

exit
