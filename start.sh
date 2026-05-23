#!/bin/bash

set -e

WORKDIR="$(pwd)"

if [ ! -d "$WORKDIR/.venv" ] && [ ! -d "$WORKDIR/venv" ]; then
    echo "Python virtual environment not found. Running setup..."
    source setup.sh
    create_python_env
fi

source $WORKDIR/.venv/bin/activate

cd $WORKDIR/server

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

echo "Starting the server..."
npm start
