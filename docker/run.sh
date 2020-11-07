docker run \
  --rm \
  -it \
  --gpus all \
  --net host \
  --ipc host \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$(pwd)/entrypoint:/entrypoint" \
  -v "$(pwd)/../ros:/ros_ws" \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  vilbert-grounding \
  "/entrypoint/entrypoint.sh"