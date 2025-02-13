LD_LIBRARY_PATH=/client_bin/:$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so CLIENT_ID=$1 CUR_SERVER_ID=$2 LOCAL_IP=$3 python $4 $5
