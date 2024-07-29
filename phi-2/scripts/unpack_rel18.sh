#! /usr/bin/env bash

DOCUMENT_DIR="data/doc"

mkdir -p $DOCUMENT_DIR


# initialize states if it does not exist
if [ -f ".states" ]; then
    true
else
    touch .states
fi

# get rel18.rar if it does not exist
if [ -f "data/rel18.rar" ]; then
    true
else
    python scripts/gdrive_download.py https://drive.google.com/file/d/1-Q8IQb_Q5lSt2Ktos8n7K2TBNXpdtKfH/view?usp=sharing --dest data
fi

if type "unrar" &> /dev/null
then
    true
else
    echo "unrar command is not available."
    echo "Installing unrar ..."
    sudo apt update
    sudo apt install unrar
fi


# extract rel18 docs
if type "unrar" &> /dev/null
then
    if ! grep rel18 .states &> /dev/null
    then
    unrar x data/rel18.rar data/doc
    echo rel18 >> .states
    echo "Done unpacking data/rel18.rar into data/doc/rel18"
    else
    echo "Unpacked data/rel18.rar into data/doc/rel18"
    fi
fi